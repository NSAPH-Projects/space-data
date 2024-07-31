import logging
import os
from glob import glob
import tarfile

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from autogluon.tabular import TabularDataset, TabularPredictor
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from scipy.interpolate import BSpline

import utils


@hydra.main(config_path="conf", config_name="train_spaceenv", version_base=None)
def main(cfg: DictConfig):
    """
    Trains a model using AutoGluon and save the results.
    """
    # == Load config ===
    spaceenv = cfg.spaceenv
    training_cfg = spaceenv.autogluon
    output_dir = "."  # hydra automatically sets this for this script since
    # the config option hydra.job.chdir is true
    # we need to set the original workind directory to the
    # this is convenient since autogluon saves the intermediate
    # models on the working directory without obvious way to change it

    # == models to train, autogluon expects a dict from name to empty dict == 
    hpars = {m: {} for m in spaceenv.ensemble_models}

    # === Data preparation ===
    # set seed
    seed_everything(spaceenv.seed)

    # == Read collection graph/data ==
    original_cwd = hydra.utils.get_original_cwd()
    collection_path = f"{original_cwd}/data_collections/{spaceenv.collection}/"
    graph_path = f"{original_cwd}/data_collections/{spaceenv.collection}/"

    # data
    logging.info(f"Reading data collection from {collection_path}:")
    data_file = glob(f"{collection_path}/data*")[0]

    if data_file.endswith("tab"):
        df_read_opts = {
            "sep": "\t",
            "index_col": spaceenv.index_col,
            "dtype": {spaceenv.index_col: str},
        }
        df = pd.read_csv(data_file, **df_read_opts)

    elif data_file.endswith("parquet"):
        df = pd.read_parquet(data_file)

    else:
        raise ValueError(f"Unknown file extension in {data_file}.")

    # remove duplicate indices
    dupl = df.index.duplicated(keep="first")
    if dupl.sum() > 0:
        logging.info(f"Removed {dupl.sum()}/{df.shape[0]} duplicate indices.")
        df = df[~dupl]

    tmp = df.shape[0]
    df = df[df[spaceenv.treatment].notna()]
    logging.info(f"Removed {tmp - df.shape[0]}/{tmp} rows with missing treatment.")

    # graph
    logging.info(f"Reading graph from {graph_path}.")
    graph_file = glob(f"{graph_path}/graph*")[0]

    # deal with possible extensions for the graph
    if graph_file.endswith(("graphml", "graphml.gz")):
        graph = nx.read_graphml(graph_file)

    elif graph_file.endswith("tar.gz"):
        with tarfile.open(graph_file, "r:gz") as tar:
            # list files in tar
            tar_files = tar.getnames()

            edges = pd.read_parquet(tar.extractfile("graph/edges.parquet"))
            coords = pd.read_parquet(tar.extractfile("graph/coords.parquet"))

        graph = nx.Graph()
        graph.add_nodes_from(coords.index)
        graph.add_edges_from(edges.values)

    else:
        raise ValueError(f"Unknown file extension of file {graph_file}.")

    # === Read covariate groups ===
    if spaceenv.covariates is not None:
        # use specified covar groups
        covar_groups = OmegaConf.to_container(spaceenv.covariates)
        covariates = utils.unpack_covariates(covar_groups)
    else:
        # assume all columns are covariates, each covariate is a group
        covar_groups = df.columns.difference([spaceenv.treatment, spaceenv.outcome])
        covar_groups = covar_groups.tolist()
        covariates = covar_groups
        spaceenv.covariates = covariates
    # d = len(covariates)

    # maintain only covariates, treatment, and outcome
    tmp = df.shape[1]
    df = df[[spaceenv.treatment] + covariates + [spaceenv.outcome]]
    logging.info(f"Removed {tmp - df.shape[1]}/{tmp} columns due to covariate choice.")

    # === Apply data transformations ===
    logging.info(f"Transforming data.")

    for tgt in ["treatment", "outcome", "covariates"]:
        scaling = getattr(spaceenv.scaling, tgt)
        transform = getattr(spaceenv.transforms, tgt)
        varnames = getattr(spaceenv, tgt)
        if tgt != "covariates":
            varnames = [varnames]
        for varname in varnames:
            if scaling is not None:
                if isinstance(scaling, (dict, DictConfig)):
                    scaling_ = scaling[varname]
                else:
                    scaling_ = scaling
                logging.info(f"Scaling {varname} with {scaling_}")
                df[varname] = utils.scale_variable(df[varname].values, scaling_)
            if transform is not None:
                if isinstance(transform, (dict, DictConfig)):
                    transform_ = transform[varname]
                else:
                    transform_ = transform
                logging.info(f"Transforming {varname} with {transform_}")
                df[varname] = utils.transform_variable(df[varname].values, transform_)

    # make treatment boolean if only two values
    if df[spaceenv.treatment].nunique() == 2:
        df[spaceenv.treatment] = df[spaceenv.treatment].astype(bool)
        is_binary_treatment = True
    else:
        # if not binary, remove bottom and top for stable training
        if spaceenv.treatment_quantile_valid_range is not None:
            fmin = 100 * spaceenv.treatment_quantile_valid_range[0]
            fmax = 100 * (1 - spaceenv.treatment_quantile_valid_range[1])
            logging.info(
                f"Removing bottom {fmin:.1f}% and top {fmax:.1f}% of treat. values for stability."
            )
            t = df[spaceenv.treatment].values
            quants = np.nanquantile(t, spaceenv.treatment_quantile_valid_range)
            df = df[(t >= quants[0]) & (t <= quants[1])]
            is_binary_treatment = False
    
    # also remove extreme values for outcome
    if spaceenv.outcome_quantile_valid_range is not None:
        fmin = 100 * spaceenv.outcome_quantile_valid_range[0]
        fmax = 100 * (1 - spaceenv.outcome_quantile_valid_range[1])
        logging.info(
            f"Removing bottom {fmin:.1f}% and top {fmax:.1f}% of outcome values for stability."
        )
        y = df[spaceenv.outcome].values
        quants = np.nanquantile(y, spaceenv.outcome_quantile_valid_range)
        df = df[(y >= quants[0]) & (y <= quants[1])]

    # === Add extra columns / techniques for better causal effect estimation
    # based on increasing attention to the treatment ===
    if not is_binary_treatment and spaceenv.bsplines:
        logging.info(f"Boosting treatment with b-splines of pctile (cont. treatment).")
        b_deg = spaceenv.bsplines_degree
        b_df = spaceenv.bsplines_df

        t = df[spaceenv.treatment].values
        t_vals = np.sort(np.unique(t))

        def get_t_pct(t):
            return np.searchsorted(t_vals, t) / len(t_vals)

        knots = np.linspace(0, 1, b_df)[1:-1].tolist()
        knots = [0] * b_deg + knots + [1] * b_deg
        spline_basis = [
            BSpline.basis_element(knots[i : (i + b_deg + 2)])
            for i in range(len(knots) - b_deg - 1)
        ]
        extra_colnames = [f"splines_{i}" for i in range(len(spline_basis))]
        extra_cols = np.stack([s(get_t_pct(t)) for s in spline_basis], axis=1)
        extra_cols = pd.DataFrame(extra_cols, columns=extra_colnames, index=df.index)
        df = pd.concat([df, extra_cols], axis=1)
    elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
        logging.info(f"Boosting treatment adding interactions with all covariates.")
        t_ind = df[spaceenv.treatment].values[:, None].astype(float)
        interacted = df[covariates].values * t_ind
        extra_colnames = [f"{c}_interact" for c in covariates]
        extra_cols = pd.DataFrame(interacted, columns=extra_colnames, index=df.index)
        df = pd.concat([df, extra_cols], axis=1)

    # test with a subset of the data
    if cfg.debug_subsample is not None:
        logging.info(f"Subsampling since debug_subsample={cfg.debug_subsample}.")
        ix = np.random.choice(range(df.shape[0]), cfg.debug_subsample, replace=False)
        df = df.iloc[ix]

    # === Harmonize data and graph ===
    intersection = set(df.index).intersection(set(graph.nodes))
    n = len(intersection)
    perc = 100 * n / len(df)
    logging.info(f"Homegenizing data and graph")
    logging.info(f"...{perc:.2f}% of the data rows (n={n}) found in graph nodes.")
    graph = nx.subgraph(graph, intersection)
    df = df.loc[list(intersection)]

    # obtain final edge list
    node2ix = {n: i for i, n in enumerate(df.index)}
    edge_list = np.array([(node2ix[e[0]], node2ix[e[1]]) for e in graph.edges])

    # fill missing if needed
    if spaceenv.fill_missing_covariate_values:
        for c in covariates:
            col_vals = df[c].values
            frac_missing = np.isnan(col_vals).mean()
            logging.info(f"Filling {100 * frac_missing:.2f}% missing values for {c}.")
            nbrs_means = utils.get_nbrs_means(col_vals, edge_list)
            col_vals[np.isnan(col_vals)] = nbrs_means[np.isnan(col_vals)]
            df[c] = col_vals

    # remove nans in outcome
    outcome_nans = np.isnan(df[spaceenv.outcome])
    logging.info(f"Removing {outcome_nans.sum()} for training since missing outcome.")
    dftrain = df[~np.isnan(df[spaceenv.outcome])]
    train_data = TabularDataset(dftrain)

    # == Spatial Train/Test Split ===
    tuning_nodes, buffer_nodes = utils.spatial_train_test_split(
        graph,
        init_frac=spaceenv.spatial_tuning.init_frac,
        levels=spaceenv.spatial_tuning.levels,
        buffer=spaceenv.spatial_tuning.buffer,
    )

    tuning_data = TabularDataset(dftrain[dftrain.index.isin(tuning_nodes)])
    train_data = TabularDataset(dftrain[~dftrain.index.isin(buffer_nodes)])
    tunefrac = 100 * len(tuning_nodes) / df.shape[0]
    trainfrac = 100 * len(train_data) / df.shape[0]
    logging.info(f"...{tunefrac:.2f}% of the rows used for tuning split.")
    logging.info(f"...{trainfrac:.2f}% of the rows used for training.")

    # === Model fitting ===
    logging.info(f"Fitting model to outcome variable on train split.")
    trainer = TabularPredictor(label=spaceenv.outcome)

    predictor = trainer.fit(
        train_data,
        **training_cfg.fit,
        tuning_data=tuning_data,
        use_bag_holdout=True,
        hyperparameters=hpars,
    )
    results = predictor.fit_summary()
    logging.info(f"Model fit summary:\n{results['leaderboard']}")

    # === Retrain on full data for the final model
    logging.info(f"Fitting to full data.")
    predictor.refit_full()

    mu = predictor.predict(df)
    mu.name = mu.name + "_pred"

    # sythetic outcome
    logging.info(f"Generating synthetic residuals for synthetic outcome.")
    mu_synth = predictor.predict(df)
    residuals = (df[spaceenv.outcome] - mu_synth).values
    synth_residuals = utils.generate_noise_like(residuals, edge_list)
    Y_synth = predictor.predict(df) + synth_residuals
    Y_synth.name = "Y_synth"

    scale = np.std(Y_synth)

    residual_smoothness = utils.moran_I(residuals, edge_list)
    synth_residual_smoothness = utils.moran_I(synth_residuals, edge_list)
    residual_nbrs_corr = utils.get_nbrs_corr(residuals, edge_list)
    synth_residual_nbrs_corr = utils.get_nbrs_corr(synth_residuals, edge_list)

    # === Counterfactual generation ===
    logging.info(f"Generating counterfactual predictions and adding residuals")
    A = df[spaceenv.treatment]
    amin, amax = np.nanmin(A), np.nanmax(A)
    n_treatment_values = len(np.unique(A))
    n_bins = min(spaceenv.treatment_max_bins, n_treatment_values)
    avals = np.linspace(amin, amax, n_bins)

    mu_cf = []
    for a in avals:
        cfdata = df.copy()
        cfdata[spaceenv.treatment] = a
        # evaluate bspline basis on treatment a fixed
        if not is_binary_treatment and spaceenv.bsplines:
            t_a_pct = np.full((n,), get_t_pct(a))
            extra_cols = np.stack([s(t_a_pct) for s in spline_basis], axis=1)
            cfdata[extra_colnames] = extra_cols
        elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
            extra_cols = df[covariates].values * a
            cfdata[extra_colnames] = extra_cols

        cfdata = TabularDataset(cfdata)
        predicted = predictor.predict(cfdata)
        mu_cf.append(predicted)
    mu_cf = pd.concat(mu_cf, axis=1)
    mu_cf.columns = [
        f"{spaceenv.outcome}_pred_{i:02d}" for i in range(len(mu_cf.columns))
    ]
    Y_cf = mu_cf + synth_residuals[:, None]
    Y_cf.columns = [f"Y_synth_{i:02d}" for i in range(len(mu_cf.columns))]

    # model leaderboard from autogluon results
    results["leaderboard"].to_csv(f"{output_dir}/leaderboard.csv", index=False)

    logging.info("Plotting counterfactuals and residuals.")
    ix = np.random.choice(len(df), cfg.num_plot_samples)
    cfpred_sample = mu_cf.iloc[ix].values
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(avals, cfpred_sample.T, color="gray", alpha=0.2)
    ax.scatter(A.iloc[ix], mu.iloc[ix], color="red")

    # Draw a line for the ATE
    ax.plot(
        avals,
        mu_cf.mean(),
        color="red",
        linestyle="--",
        label="Average Treatment Effect",
        alpha=0.5,
    )
    ax.legend()

    ax.set_xlabel(spaceenv.treatment)
    ax.set_ylabel(spaceenv.outcome)
    ax.set_title("Counterfactuals")
    fig.savefig(f"{output_dir}/counterfactuals.png", dpi=300, bbox_inches="tight")

    logging.info("Plotting histogram of true and synthetic residuals.")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(residuals, bins=20, density=True, alpha=0.5, label="True")
    ax.hist(synth_residuals, bins=20, density=True, alpha=0.5, label="Synthetic")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Residuals")
    ax.legend()
    fig.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches="tight")

    # === Compute feature importance ===
    logging.info(f"Computing feature importance.")
    featimp = predictor.feature_importance(
        train_data,
        **training_cfg.feat_importance,
    )
    # convert the .importance column to dict
    featimp = dict(featimp.importance)

    if not is_binary_treatment and spaceenv.bsplines:
        # this is the case when we want to merge the scores of all splines
        # of the treat  ment into a single score. We can use max aggregation
        tname = spaceenv.treatment
        for c in extra_colnames:
            featimp[tname] = max(featimp.get(tname, 0.0), featimp.get(c, 0.0))
            if c in featimp:
                featimp.pop(c)
    elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
        # this is the case when we want to merge interacted covariates
        # with the treatment. We can use max aggregation strategy.
        for c in covariates:
            featimp[c] = max(featimp.get(c, 0.0), featimp.get(c + "_interact", 0.0))
            if c + "_interact" in featimp:
                featimp.pop(c + "_interact")

    # yscale = np.nanstd(df[spaceenv.outcome])
    # replace with synthetic outcome standard deviation
    # yscale = np.nanstd(Y_synth)
    treat_imp = featimp[spaceenv.treatment]
    featimp = {c: float(featimp.get(c, 0.0)) / scale for c in covariates}
    featimp["treatment"] = treat_imp

    # === Fitting model to treatment variable for confounding score ===
    logging.info(f"Fitting model to treatment variable for importance score.")
    treat_trainer = TabularPredictor(label=spaceenv.treatment)
    cols = covariates + [spaceenv.treatment]
    treat_tuning_data = TabularDataset(dftrain[dftrain.index.isin(tuning_nodes)][cols])
    treat_train_data = TabularDataset(dftrain[~dftrain.index.isin(buffer_nodes)][cols])
    treat_predictor = treat_trainer.fit(
        treat_train_data,
        **training_cfg.fit,
        tuning_data=treat_tuning_data,
        use_bag_holdout=True,
        hyperparameters=hpars,
    )
    treat_predictor.refit_full()

    # normalize feature importance by scale
    tscale = np.nanstd(df[spaceenv.treatment])
    treat_featimp = treat_predictor.feature_importance(
        treat_train_data, **training_cfg.feat_importance
    )
    treat_featimp = dict(treat_featimp.importance)

    # do the reduction for the case of interactions
    if is_binary_treatment and spaceenv.binary_treatment_iteractions:
        for c in covariates:
            treat_featimp[c] = max(
                treat_featimp.get(c, 0.0), treat_featimp.get(c + "_interact", 0.0)
            )
            if c + "_interact" in treat_featimp:
                treat_featimp.pop(c + "_interact")

    treat_featimp = {c: float(treat_featimp.get(c, 0.0)) / tscale for c in covariates}

    # legacy confounding score by inimum
    cs_minimum = {k: min(treat_featimp[k], featimp[k]) for k in covariates}
    logging.info(f"Legacy conf. score by minimum:\n{cs_minimum}")

    # === Compute confounding scores ===
    # The strategy for confounding scores is to compute various types
    # using the baseline model.

    # For continous treatment compute the ERF and ITE scores
    # For categorical treatmetn additionally compute the ATE score
    # For both also use the minimum of the treatment and outcome model
    # As in the first version of the paper.

    # For comparability across environments, we divide the scores by the
    # variance of the synthetic outcome.

    # Obtain counterfactuals for the others
    cs_erf = {}
    cs_ite = {}
    cs_ate = {}  # will be empty if not binary

    for i, g in enumerate(covar_groups):
        key_ = list(g.keys())[0] if isinstance(g, dict) else g
        value_ = list(g.values())[0] if isinstance(g, dict) else [g]
        cols = dftrain.columns.difference(value_)
        leave_out_predictor = TabularPredictor(label=spaceenv.outcome)
        leave_out_predictor = leave_out_predictor.fit(
            train_data[cols],
            **spaceenv.autogluon.leave_out_fit,
            tuning_data=tuning_data[cols],
            use_bag_holdout=True,
            hyperparameters=hpars,
        )
        leave_out_predictor.refit_full()

        leave_out_mu_cf = []
        for a in avals:
            cfdata = df[cols].copy()
            cfdata[spaceenv.treatment] = a

            if not is_binary_treatment and spaceenv.bsplines:
                t_a_pct = np.full((n,), get_t_pct(a))
                extra_cols = np.stack([s(t_a_pct) for s in spline_basis], axis=1)
                cfdata[extra_colnames] = extra_cols
            elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
                extra_cols = df[covariates].values * a
                cfdata[extra_colnames] = extra_cols

            predicted = leave_out_predictor.predict(TabularDataset(cfdata))
            leave_out_mu_cf.append(predicted)
        leave_out_mu_cf = pd.concat(leave_out_mu_cf, axis=1)

        logging.info(f"[{i + 1} / {len(covar_groups)}]: {key_}")

        # compute loss normalized by the variance of the outcome
        cf_err = (leave_out_mu_cf.values - mu_cf.values) / scale
        cs_ite[key_] = float(np.sqrt((cf_err**2).mean(0)).mean())
        logging.info(f"ITE: {cs_ite[key_]:.3f}")

        erf_err = (leave_out_mu_cf.values - mu_cf.values).mean(0) / scale
        cs_erf[key_] = float(np.abs(erf_err).mean())
        logging.info(f"ERF: {cs_erf[key_]:.3f}")

        if n_treatment_values == 2:
            cs_ate[key_] = np.abs(erf_err[1] - erf_err[0])
            logging.info(f"ATE: {cs_ate[key_]:.3f}")

    # === Compute the spatial smoothness of each covariate
    logging.info(f"Computing spatial smoothness of each covariate.")
    moran_I_values = {}
    for c in covariates:
        moran_I_values[c] = utils.moran_I(df[c].values, edge_list)

    # === Save results ===
    logging.info(f"Saving synthetic data, graph, and metadata")
    X = df[df.columns.difference([spaceenv.outcome, spaceenv.treatment])]
    dfout = pd.concat([A, X, mu, mu_cf, Y_synth, Y_cf], axis=1)

    # whens saving synthetic data, respect the original data format
    if data_file.endswith("tab"):
        dfout.to_csv(f"{output_dir}/synthetic_data.tab", sep="\t", index=True)
    elif data_file.endswith("parquet"):
        dfout.to_parquet(f"{output_dir}/synthetic_data.parquet")

    # save subgraph in the right format
    if graph_file.endswith(("graphml", "graphml.gz")):
        ext = "graphml.gz" if graph_file.endswith("graphml.gz") else "graphml"
        tgt_graph_path = f"{output_dir}/graph.{ext}"
        nx.write_graphml(graph, tgt_graph_path)

    elif graph_file.endswith("tar.gz"):
        # save edges and coords
        edges = pd.DataFrame(np.array(list(graph.edges)), columns=["source", "target"])
        coords = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        # save again as a tar.gz
        with tarfile.open(f"{output_dir}/graph.tar.gz", "w:gz") as tar:
            os.makedirs("graph", exist_ok=True)
            edges.to_parquet("graph/edges.parquet")
            coords.to_parquet("graph/coords.parquet")
            tar.add("graph/")

    metadata = {
        "base_name": f"{spaceenv.base_name}",
        "treatment": spaceenv.treatment,
        "predicted_outcome": spaceenv.outcome,
        "synthetic_outcome": "Y_synth",
        "confounding_score": utils.sort_dict(cs_minimum),
        "confounding_score_erf": utils.sort_dict(cs_erf),
        "confounding_score_ite": utils.sort_dict(cs_ite),
        "confounding_score_ate": utils.sort_dict(cs_ate),
        "spatial_scores": utils.sort_dict(moran_I_values),
        "outcome_importance": utils.sort_dict(featimp),
        "treatment_importance": utils.sort_dict(treat_featimp),
        "covariates": list(covariates),
        "treatment_values": avals.tolist(),
        "covariate_groups": covar_groups,
        "original_residual_spatial_score": float(residual_smoothness),
        "synthetic_residual_spatial_score": float(synth_residual_smoothness),
        "original_nbrs_corr": float(residual_nbrs_corr),
        "synthetic_nbrs_corr": float(synth_residual_nbrs_corr),
    }

    # save metadata and resolved config
    with open(f"{output_dir}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f, sort_keys=False)


if __name__ == "__main__":
    main()

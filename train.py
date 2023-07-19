import logging
import os

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from autogluon.tabular import TabularDataset, TabularPredictor
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from utils import (
    download_dataverse_data,
    generate_noise_like,
    moran_I,
    scale_variable,
    transform_variable,
    sort_dict,
    spatial_train_test_split,
    unpack_covariates,
)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Trains a model using AutoGluon and save the results.
    """
    # == Load config ===
    spaceenv = cfg.spaceenv
    training_cfg = spaceenv.autogluon

    # === Data preparation ===
    # set seed
    seed_everything(spaceenv.seed)

    # root directory
    owd = get_original_cwd()

    # download dataset and graph from dataverse if necessary
    files = {"Graph": spaceenv.graph_path, "Data": spaceenv.data_path}
    for obj, path in files.items():
        if not os.path.exists(f"{owd}/{path}"):
            filename = os.path.basename(path)
            logging.info(f"{obj} not found. Downloading from dataverse.")
            download_dataverse_data(
                filename=filename,
                dataverse_baseurl=cfg.dataverse.baseurl,
                dataverse_pid=cfg.dataverse.pid,
                output_dir=f"{owd}/data",
            )

    # load data
    logging.info(f"Loading data from {spaceenv.data_path}")
    data_ext = spaceenv.data_path.split(".")[-1]
    delim = "," if data_ext == "csv" else "\t"
    df_read_opts = {"sep": delim}
    if spaceenv.index_col is not None:
        df_read_opts["index_col"] = spaceenv.index_col
        df_read_opts["dtype"] = {spaceenv.index_col: str}
    df = pd.read_csv(f"{owd}/{spaceenv.data_path}", **df_read_opts)
    df = df[df[spaceenv.treatment].notna()]

    if spaceenv.covariates is not None:
        # use specified covar groups
        covar_groups = OmegaConf.to_container(spaceenv.covariates)
        covariates = unpack_covariates(covar_groups)
    else:
        # assume all columns are covariates, each covariate is a group
        covar_groups = df.columns.difference([spaceenv.treatment, spaceenv.outcome])
        covar_groups = covar_groups.tolist()
        covariates = covar_groups

    d = len(covariates)

    # maintain only covariates, treatment, and outcome
    df = df[[spaceenv.treatment] + covariates + [spaceenv.outcome]]

    # get treatment and outcome and
    # apply transforms to treatment or outcome if needed
    # TODO: improve the syntax for transforms with covariates
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
                df[varname] = scale_variable(df[varname].values, scaling_)
            if transform is not None:
                if isinstance(transform, (dict, DictConfig)):
                    transform_ = transform[varname]
                else:
                    transform_ = transform
                logging.info(f"Transforming {varname} with {transform_}")
                df[varname] = transform_variable(df[varname].values, transform_)

    # make treatment boolean if only two values
    if df[spaceenv.treatment].nunique() == 2:
        df[spaceenv.treatment] = df[spaceenv.treatment].astype(bool)

    # repeat the treatment column several times to allow for the probability
    # of the ML autogluon models using the treatment as a feature
    # Mathematically, it doesn't change the model space, but avoids the
    # treatment effect getting lost.
    if spaceenv.boost_treatment:
        extra_cols = df[spaceenv.treatment].copy()
        extra_cols = pd.concat([extra_cols] * (d - 1), axis=1)
        extra_colnames = [f"boosted_treatment_{i:02d}" for i in range(d - 1)]
        extra_cols.columns = extra_colnames
        df = pd.concat([df, extra_cols], axis=1)

    # read graphml
    logging.info(f"Reading graph from {spaceenv.graph_path}")
    graph = nx.read_graphml(f"{owd}/{spaceenv.graph_path}")

    # test with a subset of the data
    if cfg.debug_subsample is not None:
        n = cfg.debug_subsample
        ix = np.random.choice(n, 100, replace=False)
        df = df.iloc[ix]

    # keep intersection of nodes in graph and data
    intersection = set(df.index).intersection(set(graph.nodes))
    n = len(intersection)
    perc = 100 * n / len(df)
    logging.info(f"Homegenizing data and graph")
    logging.info(f"...{perc:.2f}% of the data rows (n={n}) found in graph nodes.")
    graph = nx.subgraph(graph, intersection)
    df = df.loc[intersection]

    # remove nans from training data
    dftrain = df[~np.isnan(df[spaceenv.outcome])]
    train_data = TabularDataset(dftrain)

    # == Spatial Train/Test Split ===
    tuning_nodes, buffer_nodes = spatial_train_test_split(
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
    )
    results = predictor.fit_summary()
    logging.info(f"Model fit summary:\n{results['leaderboard']}")

    # === Retrain on full data for the final model
    logging.info(f"Fitting to full data.")
    predictor.refit_full("best")

    mu = predictor.predict(df)
    mu.name = mu.name + "_pred"

    # synsthetic outcome
    logging.info(f"Generating synthetic residuals for synthetic outcome.")
    mu_synth = predictor.predict(df)
    residuals = df[spaceenv.outcome] - mu_synth
    synth_residuals = generate_noise_like(residuals, graph)
    Y_synth = predictor.predict(df) + synth_residuals
    Y_synth.name = "Y_synth"

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
        if spaceenv.boost_treatment:
            cfdata[extra_colnames] = a
        cfdata = TabularDataset(cfdata)
        predicted = predictor.predict(cfdata)
        mu_cf.append(predicted)
    mu_cf = pd.concat(mu_cf, axis=1)
    mu_cf.columns = [
        f"{spaceenv.outcome}_pred_{i:02d}" for i in range(len(mu_cf.columns))
    ]
    Y_cf = mu_cf + synth_residuals[:, None]
    Y_cf.columns = [f"Y_synth_{i:02d}" for i in range(len(mu_cf.columns))]

    # === Compute feature importance ===
    logging.info(f"Computing feature importance.")
    if spaceenv.boost_treatment:
        featimp_data = TabularDataset(dftrain.drop(extra_colnames, axis=1))
    else:
        featimp_data = train_data

    featimp = predictor.feature_importance(
        featimp_data,
        **training_cfg.feat_importance,
    )

    yscale = np.nanstd(df[spaceenv.outcome])
    featimp = (featimp / yscale).importance.to_dict()
    treat_imp = featimp[spaceenv.treatment]
    featimp = {c: float(featimp.get(c, 0.0)) for c in covariates}
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
    )
    treat_predictor.refit_full("best")

    # normalize feature importance by scale
    tscale = np.nanstd(df[spaceenv.treatment])
    treat_featimp = treat_predictor.feature_importance(
        treat_train_data, **training_cfg.feat_importance
    )
    treat_featimp = (treat_featimp / tscale).importance.to_dict()
    treat_featimp = {c: float(treat_featimp.get(c, 0.0)) for c in covariates}

    # === Compute confounding score ===
    scoring_method = cfg.confounding_score_method
    logging.info(f"Computing confounding score using {scoring_method}.")

    if scoring_method == "feat_importance":
        # compute the confounding score
        confounding_score = {k: min(treat_featimp[k], featimp[k]) for k in covariates}
        logging.info(f"Combined importances with min:\n{confounding_score}")

    elif scoring_method == "leave_out_fit":
        confounding_score = {}

        for i, g in enumerate(covar_groups):
            key_ = list(g.keys())[0] if isinstance(g, dict) else g
            value_ = list(g.values())[0] if isinstance(g, dict) else [g]
            cols = dftrain.columns.difference(value_)
            leave_out_predictor = TabularPredictor(label=spaceenv.outcome)
            leave_out_predictor = leave_out_predictor.fit(
                train_data[cols],
                **spaceenv.autogluon.fit,
                tuning_data=tuning_data[cols],
                use_bag_holdout=True,
            )
            leave_out_predictor.refit_full("best")

            leave_out_mu_cf = []
            for a in avals:
                cfdata = df[cols].copy()
                cfdata[spaceenv.treatment] = a
                predicted = leave_out_predictor.predict(TabularDataset(cfdata))
                leave_out_mu_cf.append(predicted)
            leave_out_mu_cf = pd.concat(leave_out_mu_cf, axis=1)

            # compute loss normalized by the variance of the outcome
            rss = np.mean((leave_out_mu_cf.values - mu_cf.values) ** 2)
            # score = rss / mu_cf.values.var()
            score = rss / yscale
            confounding_score[key_] = score
            logging.info(f"[{i + 1} / {len(covar_groups)}] {key_}: {score:.4f}")

    else:
        raise ValueError(f"Unrecognized confounding score method: {scoring_method}")

    # === Compute the spatial smoothness of each covariate
    logging.info(f"Computing spatial smoothness of each covariate.")
    moran_I_values = {}
    adjmat = nx.adjacency_matrix(graph, nodelist=df.index).toarray()
    for c in covariates:
        moran_I_values[c] = moran_I(df[c], adjmat)

    # === Save results ===
    logging.info(f"Saving synthetic data, graph, and metadata")
    X = df[df.columns.difference([spaceenv.outcome, spaceenv.treatment])]
    dfout = pd.concat([A, X, mu, mu_cf, Y_synth, Y_cf], axis=1)
    dfout.to_csv("synthetic_data.csv")
    nx.write_graphml(graph, "graph.graphml")

    name_prefix = "spaceb" if n_bins == 2 else "spacec"
    metadata = {
        "name": f"{name_prefix}_{spaceenv.base_name}",
        "treatment": spaceenv.treatment,
        "predicted_outcome": spaceenv.outcome,
        "synthetic_outcome": "Y_synth",
        "confounding_score": sort_dict(confounding_score),
        "spatial_scores": sort_dict(moran_I_values),
        "outcome_importance": sort_dict(featimp),
        "treatment_importance": sort_dict(treat_featimp),
        "covariates": list(covariates),
        "treatment_values": avals.tolist(),
        "covariate_groups": covar_groups,
    }

    with open("metadata.yaml", "w") as f:
        yaml.dump(metadata, f, sort_keys=False)

    # model leaderboard from autogluon results
    results["leaderboard"].to_csv("leaderboard.csv", index=False)

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
    fig.savefig("counterfactuals.png", dpi=300, bbox_inches="tight")

    logging.info("Plotting histogram of true and synthetic residuals.")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(residuals, bins=20, density=True, alpha=0.5, label="True")
    ax.hist(synth_residuals, bins=20, density=True, alpha=0.5, label="Synthetic")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Residuals")
    ax.legend()
    fig.savefig("residuals.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

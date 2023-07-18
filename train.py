import os
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import logging
import yaml
import hydra
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from autogluon.tabular import TabularDataset, TabularPredictor

from utils import (
    transform_variable,
    scale_variable,
    generate_noise_like,
    moran_I,
    download_dataverse_data,
    sort_dict,
)

logger = logging.getLogger(__name__)


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
            logger.info(f"{obj} not found. Downloading from dataverse.")
            download_dataverse_data(
                filename=filename,
                dataverse_baseurl=cfg.dataverse.baseurl,
                dataverse_pid=cfg.dataverse.pid,
                output_dir=f"{owd}/data",
            )

    # load data
    logger.info(f"Loading data from {spaceenv.data_path}")
    data_ext = spaceenv.data_path.split(".")[-1]
    delim = "," if data_ext == "csv" else "\t"
    df_read_opts = {"sep": delim}
    if spaceenv.index_col is not None:
        df_read_opts["index_col"] = spaceenv.index_col
        df_read_opts["dtype"] = {spaceenv.index_col: str}
    df = pd.read_csv(f"{owd}/{spaceenv.data_path}", **df_read_opts)
    df = df[df[spaceenv.treatment].notna()]

    # maintain only covariates, treatment, and outcome
    if spaceenv.covariates is not None:
        covariates = spaceenv.covariates
    else:
        covariates = list(df.columns.difference([spaceenv.treatment, spaceenv.outcome]))
    d = len(covariates)
    df = df[[spaceenv.treatment] + covariates + [spaceenv.outcome]]

    # get treatment and outcome and
    # apply transforms to treatment or outcome if needed
    # TODO: improve the syntax for transforms with covariates
    logger.info(f"Transforming data.")
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
                logger.info(f"Scaling {varname} with {scaling_}")
                df[varname] = scale_variable(df[varname].values, scaling_)
            if transform is not None:
                if isinstance(transform, (dict, DictConfig)):
                    transform_ = transform[varname]
                else:
                    transform_ = transform
                logger.info(f"Transforming {varname} with {transform_}")
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
    logger.info(f"Reading graph from {spaceenv.graph_path}")
    graph = nx.read_graphml(f"{owd}/{spaceenv.graph_path}")

    # test with a subset of the data
    if cfg.debug_subsample is not None:
        n = cfg.debug_subsample
        ix = np.random.choice(n, 100, replace=False)
        df = df.iloc[ix]

    # keep intersectio of nodes in graph and data
    intersection = set(df.index).intersection(set(graph.nodes))
    n = len(intersection)
    perc = 100 * n / len(df)
    logger.info(f"Homegenizing data and graph")
    logger.info(f"...{perc:.2f}% of the data rows (n={n}) found in graph nodes.")
    graph = nx.subgraph(graph, intersection)
    df = df.loc[intersection]

    # remove nans from training data
    dftrain = df[~np.isnan(df[spaceenv.outcome])]
    train_data = TabularDataset(dftrain)

    # == Spatial Train/Test Split ===
    if spaceenv.spatial_tuning.frac > 0:
        levels = spaceenv.spatial_tuning.levels
        buffer_size = spaceenv.spatial_tuning.buffer
        logger.info(f"Selecting tunning split removing {levels} nbrs from val. pts.")

        # make dict of neighbors from graph
        node_list = np.array(graph.nodes())
        nbrs = {node: set(graph.neighbors(node)) for node in node_list}

        # first find the centroid of the tuning subgraph
        num_tuning_centroids = int(spaceenv.spatial_tuning.frac * df.shape[0])
        tuning_nodes = np.random.choice(
            df.shape[0], size=num_tuning_centroids, replace=False
        )
        tuning_nodes = set(node_list[tuning_nodes])

        # not remove all neighbors of the tuning centroids from the training data
        for _ in range(levels):
            tmp = tuning_nodes.copy()
            for node in tmp:
                for nbr in nbrs[node]:
                    tuning_nodes.add(nbr)
        tuning_nodes = list(tuning_nodes)

        # buffer
        buffer_nodes = set(tuning_nodes.copy())
        for _ in range(buffer_size):
            tmp = buffer_nodes.copy()
            for node in tmp:
                for nbr in nbrs[node]:
                    buffer_nodes.add(nbr)
        buffer_nodes = list(set(buffer_nodes))

        # split data into tuning and trainingå
        tuning_data = TabularDataset(dftrain[dftrain.index.isin(tuning_nodes)])
        train_data = TabularDataset(dftrain[~dftrain.index.isin(buffer_nodes)])
        tunefrac = 100 * len(tuning_nodes) / df.shape[0]
        trainfrac = 100 * len(train_data) / df.shape[0]
        logger.info(f"...{tunefrac:.2f}% of the rows used for tuning split.")
        logger.info(f"...{trainfrac:.2f}% of the rows used for training.")
    else:
        tuning_data = None

    # === Model fitting ===
    logger.info(f"Fitting model to outcome variable.")
    trainer = TabularPredictor(label=spaceenv.outcome)
    predictor = trainer.fit(
        train_data,
        **training_cfg.fit,
        tuning_data=tuning_data,
        use_bag_holdout=(spaceenv.spatial_tuning.frac > 0),
    )
    featimp = predictor.feature_importance(train_data, **training_cfg.feat_importance)

    results = predictor.fit_summary()
    mu = predictor.predict(df)
    mu.name = mu.name + "_pred"

    # inject noise
    logger.info(f"Generating synthetic residuals for synthetic outcome.")
    mu_synth = predictor.predict(df)
    residuals = df[spaceenv.outcome] - mu_synth
    synth_residuals = generate_noise_like(residuals, graph)
    Y_synth = predictor.predict(df) + synth_residuals
    Y_synth.name = "Y_synth"

    logger.info(f"Fitting model to treatment variable for confounding score.")
    treat_trainer = TabularPredictor(label=spaceenv.treatment)
    treat_train_data = TabularDataset(dftrain[covariates + [spaceenv.treatment]])
    treat_predictor = treat_trainer.fit(treat_train_data, **training_cfg.fit)

    # normalize feature importance by scale
    yscale = np.nanstd(df[spaceenv.outcome])
    tscale = np.nanstd(df[spaceenv.treatment])
    treat_featimp = treat_predictor.feature_importance(
        treat_train_data, **training_cfg.feat_importance
    )
    treat_featimp = (treat_featimp / tscale).importance.to_dict()
    treat_featimp = {c: float(treat_featimp.get(c, 0.0)) for c in covariates}

    # compute the specific importance of the treatment on the outcome model
    outcome_featimp = (featimp / yscale).importance.to_dict()
    outcome_featimp = {c: float(outcome_featimp.get(c, 0.0)) for c in covariates}

    treat_importance_outcome_model = featimp.importance.loc[spaceenv.treatment]
    if spaceenv.boost_treatment:
        extra_importances = featimp.importance.loc[extra_colnames].to_list()
        treat_importances_all = [treat_importance_outcome_model] + extra_importances
        treat_importance_outcome_model = np.mean(treat_importances_all)
    outcome_featimp["treatment"] = float(treat_importance_outcome_model)

    # compute the confounding score
    confounding_score = {k: min(treat_featimp[k], outcome_featimp[k]) for k in covariates}

    # === Counterfactual generation ===
    logger.info(f"Generating counterfactual predictions and adding residuals")
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

    # === Compute the spatial smoothness of each covariate
    logger.info(f"Computing spatial smoothness of each covariate.")
    moran_I_values = {}
    adjmat = nx.adjacency_matrix(graph, nodelist=df.index).toarray()
    for c in covariates:
        moran_I_values[c] = moran_I(df[c], adjmat)

    # === Save results ===
    logger.info(f"Saving synthetic data, graph, and metadata")
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
        "outcome_importance": sort_dict(outcome_featimp),
        "treatment_importance": sort_dict(treat_featimp),
        "covariates": list(covariates),
        "treatment_values": list(avals),
    }
    with open("metadata.yaml", "w") as f:
        yaml.dump(metadata, f, sort_keys=False)

    # model leaderboard from autogluon results
    results["leaderboard"].to_csv("leaderboard.csv", index=False)

    logger.info("Plotting counterfactuals and residuals.")
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

    logger.info("Plotting histogram of true and synthetic residuals.")
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

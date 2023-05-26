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


from utils import transform_variable, scale_variable, generate_noise_like, moran_I

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Trains a model using AutoGluon and save the results.
    """
    # === Data preparation ===
    # set seed
    seed_everything(cfg.seed)

    # load data
    logger.info(f"Loading data from {cfg.data.data_path}")
    owd = get_original_cwd()
    df_read_opts = {}
    if cfg.data.index_col is not None:
        df_read_opts["index_col"] = cfg.data.index_col
        df_read_opts["dtype"] = {cfg.data.index_col: str}
    df = pd.read_csv(f"{owd}/{cfg.data.data_path}", **df_read_opts)
    df = df[df[cfg.data.treatment].notna()]

    # maintain only covariates, treatment, and outcome
    if cfg.data.covariates is not None:
        covariates = cfg.data.covariates
    else:
        covariates = list(df.columns.difference([cfg.data.treatment, cfg.data.outcome]))
    df = df[[cfg.data.treatment] + covariates + [cfg.data.outcome]]

    # read graphml
    logger.info(f"Reading graph from {cfg.data.graph_path}")
    graph = nx.read_graphml(f"{owd}/{cfg.data.graph_path}")

    # test with a subset of the data
    if cfg.debug_subsample is not None:
        n = cfg.debug_subsample
        ix = np.random.choice(n, 100, replace=False)
        df = df.iloc[ix]

    # keep intersectio of nodes in graph and data
    intersection = set(df.index).intersection(set(graph.nodes))
    perc = 100 * len(intersection) / len(df)
    logger.info(f"Homegenizing data and graph")
    logger.info(f"...{perc:.2f}% of the data rows found in graph nodes.")
    graph = nx.subgraph(graph, intersection)
    df = df.loc[intersection]

    # remove nans from training data
    dftrain = df[~np.isnan(df[cfg.data.outcome])]
    train_data = TabularDataset(dftrain)

    # get treatment and outcome and
    # apply transforms to treatment or outcome if needed
    logger.info(f"Transforming data.")
    for tgt in ["treatment", "outcome"]:
        scaling = getattr(cfg.data.scaling, tgt)
        transform = getattr(cfg.data.transforms, tgt)
        varname = getattr(cfg.data, tgt)
        if scaling is not None:
            logger.info(f"Scaling {varname} with {scaling}")
            df[varname] = scale_variable(df[varname].values, scaling)
        if transform is not None:
            logger.info(f"Transforming {varname} with {transform}")
            df[varname] = transform_variable(df[varname].values, transform)

    # === Model fitting ===
    logger.info(f"Fitting model to outcome variable.")
    trainer = TabularPredictor(label=cfg.data.outcome)
    predictor = trainer.fit(train_data, **cfg.autogluon.fit)
    featimp = predictor.feature_importance(train_data)
    results = predictor.fit_summary()
    mu = predictor.predict(df)
    mu.name = mu.name + "_pred"

    # inject noise
    logger.info(f"Generating synthetic residuals for synthetic outcome.")
    fit_residuals = dftrain[cfg.data.outcome] - predictor.predict(train_data)
    synth_residuals = generate_noise_like(fit_residuals, graph)
    Y_synth = predictor.predict(df) + synth_residuals
    Y_synth.name = "Y_synth"

    logger.info(f"Fitting model to treatment variable for confounding score.")
    treatment_trainer = TabularPredictor(label=cfg.data.treatment)
    treatment_train_data = TabularDataset(
        dftrain[dftrain.columns.difference([cfg.data.outcome])]
    )
    treatment_predictor = treatment_trainer.fit(
        treatment_train_data, **cfg.autogluon.fit
    )
    treatment_featimp = treatment_predictor.feature_importance(treatment_train_data)
    outcome_featimp = featimp.loc[treatment_featimp.index]
    confounding_score = np.minimum(
        outcome_featimp.importance, treatment_featimp.importance
    ).sort_values(ascending=False)

    # === Counterfactual generation ===
    logger.info(f"Generating counterfactual predictions and adding residuals")
    A = df[cfg.data.treatment]
    amin, amax = np.nanmin(A), np.nanmax(A)
    n_treatment_values = len(np.unique(A))
    n_bins = min(cfg.data.treatment_max_bins, n_treatment_values)
    avals = np.linspace(amin, amax, n_bins)

    mu_cf = []
    for a in avals:
        cfdata = df.copy()
        cfdata[cfg.data.treatment] = a
        cfdata = TabularDataset(cfdata)
        predicted = predictor.predict(cfdata)
        mu_cf.append(predicted)
    mu_cf = pd.concat(mu_cf, axis=1)
    mu_cf.columns = [
        f"{cfg.data.outcome}_pred_{i:02d}" for i in range(len(mu_cf.columns))
    ]
    Y_cf = mu_cf + synth_residuals[:, None]
    Y_cf.columns = [f"Y_synth_{i:02d}" for i in range(len(mu_cf.columns))]

    # === Compute the spatial smoothness of each covariate
    logger.info(f"Computing spatial smoothness of each covariate.")
    moran_I_values = {}
    adjmat = nx.adjacency_matrix(graph, nodelist=df.index).toarray()
    for c in covariates:
        moran_I_values[c] = moran_I(df[c], adjmat)
    moran_I_values = {
        k: v for k, v in sorted(moran_I_values.items(), key=lambda item: item[1], reverse=True)}

    # === Save results ===
    logger.info(f"Saving synthetic data, graph, and metadata")
    X = df[df.columns.difference([cfg.data.outcome, cfg.data.treatment])]
    dfout = pd.concat([A, X, mu, mu_cf, Y_synth, Y_cf], axis=1)
    dfout.to_csv("synthetic_data.csv")
    nx.write_graphml(graph, "graph.graphml")

    name_prefix = "spaceb" if n_bins == 2 else "spacec"
    metadata = {
        "name": f"{name_prefix}_{cfg.data.base_name}",
        "treatment": cfg.data.treatment,
        "predicted_outcome": cfg.data.outcome,
        "synthetic_outcome": "Y_synth",
        "confounding_score": confounding_score.to_dict(),
        "spatial_scores": moran_I_values,
        "covariates": list(X.columns),
        "tretment_values": avals.tolist(),
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
    ax.set_xlabel(cfg.data.treatment)
    ax.set_ylabel(cfg.data.outcome)
    ax.set_title("Counterfactuals")
    fig.savefig("counterfactuals.png", dpi=300, bbox_inches="tight")

    logger.info("Plotting histogram of true and synthetic residuals.")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(fit_residuals, bins=20, density=True, alpha=0.5, label="True")
    ax.hist(synth_residuals, bins=20, density=True, alpha=0.5, label="Synthetic")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Residuals")
    ax.legend()
    fig.savefig("residuals.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

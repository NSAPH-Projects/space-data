from omegaconf import DictConfig

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from autogluon.tabular import TabularDataset, TabularPredictor
import yaml

import hydra
from hydra.utils import get_original_cwd

from utils import transform_variable, scale_variable, generate_noise_like


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # set seed
    seed_everything(cfg.seed)

    # load data
    owd = get_original_cwd()
    df = pd.read_csv(f"{owd}/{cfg.data.path}", index_col=cfg.data.index_col)
    df = df[df[cfg.data.treatment].notna()]

    # test with a subset of the data
    ix = np.random.choice(len(df), 100)
    df = df.iloc[ix]

    # remove nans from training data
    dftrain = df[~np.isnan(df[cfg.data.outcome])]
    train_data = TabularDataset(dftrain)

    # get treatment and outcome and
    # apply transforms to treatment or outcome if needed
    for tgt in ["treatment", "outcome"]:
        scaling = getattr(cfg.data.scaling, tgt)
        transform = getattr(cfg.data.transforms, tgt)
        varname = getattr(cfg.data, tgt)
        df[varname] = scale_variable(df[varname].values, scaling)
        df[varname] = transform_variable(df[varname].values, transform)

    # train
    trainer = TabularPredictor(label=cfg.data.outcome)
    predictor = trainer.fit(train_data, **cfg.autogluon.fit)
    feature_importance = predictor.feature_importance(train_data)
    results = predictor.fit_summary()
    mu_synth = predictor.predict(df)

    # read graphml
    graph = nx.read_graphml(f"{owd}/{cfg.data.graphml}")

    # compute easy noise model with CAR
    residuals = df[cfg.data.outcome] - mu_synth
    graph_laplacian = nx.laplacian_matrix(graph).toarray()

    # get counterfactual treatments and predictions
    A = df[cfg.data.treatment]
    amin, amax = np.nanmin(A), np.nanmax(A)
    avals = np.linspace(amin, amax, cfg.data.treatment_bins)

    mu_cf = []
    for a in avals:
        cfdata = df.copy()
        cfdata[cfg.data.treatment] = a
        cfdata = TabularDataset(cfdata)
        predicted = predictor.predict(cfdata)
        mu_cf.append(predicted)
    mu_cf = pd.concat(mu_cf, axis=1)
    mu_cf.columns = [f"{cfg.data.outcome}_{i:02d}" for i in range(len(mu_cf.columns))]

    # save fit results
    X = df[df.columns.difference([cfg.data.outcome, cfg.data.treatment])]
    dfout = pd.concat([A, X, mu_synth, mu_cf], axis=1)
    dfout.to_csv("synthetic_cfg.data.csv")

    metadata = {
        "treatment": cfg.data.treatment,
        "synthetic_outcome": cfg.data.outcome,
        "covariates": list(X.columns),
        "tretment_values": avals.tolist(),
        "feature_importance": feature_importance.importance.to_dict(),
    }
    with open("metacfg.data.yaml", "w") as f:
        yaml.dump(metadata, f)
    results["leaderboard"].to_csv("leaderboard.csv", index=False)

    # plot potential outcome curves
    ix = np.random.choice(len(df), cfg.num_plot_samples)
    cfpred_sample = mu_cf.iloc[ix].values
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(avals, cfpred_sample.T, color="gray", alpha=0.2)
    ax.scatter(A.iloc[ix], mu_synth.iloc[ix], color="red")
    ax.set_xlabel(cfg.data.treatment)
    ax.set_ylabel(cfg.data.outcome)
    ax.set_title("Counterfactuals")
    fig.savefig("counterfactuals.png", dpi=300)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()

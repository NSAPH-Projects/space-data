from omegaconf import DictConfig

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from autogluon.tabular import TabularDataset, TabularPredictor
import yaml

import hydra
from hydra.utils import get_original_cwd

from utils import transform_variable, scale_variable


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    datadir = get_original_cwd() + "/data"
    data = cfg.data
    df = pd.read_csv(f"{datadir}/{data.path}", index_col=data.index_col)
    df = df[df[data.treatment].notna()]

    # remove nans from training data
    dftrain = df[~np.isnan(df[data.outcome])]
    train_data = TabularDataset(dftrain)

    # get treatmetn and outcome and
    # apply transforms to treatment or outcome if needed
    for tgt in ["treatment", "outcome"]:
        scaling = getattr(data.scaling, tgt)
        transform = getattr(data.transforms, tgt)
        varname = getattr(data, tgt)
        df[varname] = scale_variable(df[varname].values, scaling)
        df[varname] = transform_variable(df[varname].values, transform)

    # train
    trainer = TabularPredictor(label=data.outcome)
    predictor = trainer.fit(train_data, **cfg.autogluon.fit)
    feature_importance = predictor.feature_importance(train_data)
    results = predictor.fit_summary()
    Ysynth = predictor.predict(df)

    # get counterfactual treatments and predictions
    A = df[data.treatment]
    amin, amax = np.nanmin(A), np.nanmax(A)
    avals = np.linspace(amin, amax, cfg.continuous_treatment_bins)

    Ycf = []
    for a in avals:
        cfdata = df.copy()
        cfdata[data.treatment] = a
        cfdata = TabularDataset(cfdata)
        predicted = predictor.predict(cfdata)
        Ycf.append(predicted)
    Ycf = pd.concat(Ycf, axis=1)
    Ycf.columns = [f"{data.outcome}_{i:02d}" for i in range(len(Ycf.columns))]

    # save fit results
    X = df[df.columns.difference([data.outcome, data.treatment])]
    dfout = pd.concat([A, X, Ysynth, Ycf], axis=1)
    dfout.to_csv("synthetic_data.csv")

    metadata = {
        "treatment": data.treatment,
        "synthetic_outcome": data.outcome,
        "covariates": list(X.columns),
        "tretment_values": avals.tolist(),
        "feature_importance": feature_importance.importance.to_dict(),
    }
    with open("metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
    results["leaderboard"].to_csv("leaderboard.csv", index=False)

    # plot potential outcome curves
    ix = np.random.choice(len(df), cfg.num_plot_samples)
    cfpred_sample = Ycf.iloc[ix].values
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(avals, cfpred_sample.T, color="gray", alpha=0.2)
    ax.scatter(A.iloc[ix], df[data.outcome].iloc[ix], color="red")
    ax.set_xlabel(data.treatment)
    ax.set_ylabel(data.outcome)
    ax.set_title("Counterfactuals")
    fig.savefig("counterfactuals.png", dpi=300)


if __name__ == "__main__":
    matplotlib.use("Agg")
    seed_everything(42)
    main()

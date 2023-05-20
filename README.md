# SpaCE Training 🌎💪🏋️‍♀️

## Install prerequisites
```
conda env create -f requirements.yaml
conda activate spacetrain
```

## Run code
```
python train.py data=elections
```

## Results
Look at the files generated in `outputs/`:
 - `synthetic_data.csv`: data frame with all the synthetic and real data 
 - `metadata.yaml`: info about the generated data (column names, features importance, etc.)
 - `leaderboard.csv`: results from `autogluon` fit.
 - `counterfactuals.png` image with generated potential outcome curves, it gives a good idea of the confounding

## Add new datasets

Add a config file in `conf/data`. Look at `conf/data/elections.yaml` for inspiration. Look at `conf/config.yaml` the elements marked with `???` are mandatory in `conf/data`.


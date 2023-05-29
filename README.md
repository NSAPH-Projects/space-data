# SpaCE Data 🌎💪🏋️‍♀️

## Install prerequisites
```
conda env create -f requirements.yaml
conda activate spacedata
```

Creation of the conda environment is known to fail on Intel-based Macs. For these cases, you can use the Dockerfile in the repo.


## Run code

export your dataverse api token
```
export DATAVERSE_TOKEN=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

```
python train.py spaceenv=elect_dempct_college
```

## Add new datasets

Add a config file in `conf/spaceenv`. Look at `conf/spaceenv/elect_dempct_college.yaml` for inspiration. The elements marked with `???` under the `spaceenv` field in `conf/config.yaml' are mandatory.

## Results
Look at the files generated in `outputs/`:
 - `synthetic_data.csv`: data frame with all the synthetic and real data 
 - `metadata.yaml`: info about the generated data (column names, features importance, etc.)
 - `leaderboard.csv`: results from `autogluon` fit.
 - `counterfactuals.png` image with generated potential outcome curves, it gives a good idea of the confounding

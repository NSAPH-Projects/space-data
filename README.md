# SpaCE Data 🌎💪🏋️‍♀️

## Install prerequisites
```
conda env create -f requirements.yaml
conda activate spacedata
```

Creation of the conda environment is known to fail on Intel-based Macs. For these cases, you can use the Dockerfile in the repo.


## Run code

Creating a dataset consists of two steps: training and uploading.

### Training

Run the following command
```
python train.py spaceenv=<config_file_name>
```
For example, you can use `python train.py spaceenv=elect_dempct_college` to train the `elect_dempct_college` space environment. In general, `<config_file_name>` can be any of the config files in `conf/spaceenv/`. The config files are `.yaml` files that contain the parameters for the training. The `spaceenv` parameter is mandatory and it should be the name of the config file without the `.yaml` extension.

The outputs will be saved in `outputs/<base_name>` where the `base_name` is specified as a mandatory field in the config file. The outputs are:
 - `synthetic_data.csv`: data frame with all the synthetic and real data 
 - `metadata.yaml`: info about the generated data (column names, features importance, etc.)
 - `leaderboard.csv`: results from `autogluon` fit.
 - `counterfactuals.png` image with generated potential outcome curves, it gives a good idea of the confounding

**⚠️Note⚠️**: The recommended pattern is that the name of the config file and basenames are the same when pushing a file to the repository. In the future this may be automatic.


### Uploading

You will need a Harvard Dataverse API token and export it as an environment variable. Then run

```
export DATAVERSE_TOKEN=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python upload.py base_name=<base_name>
```



## Add new datasets

Add a config file in `conf/spaceenv`. Look at `conf/spaceenv/elect_dempct_college.yaml` for inspiration. The elements marked with `???` under the `spaceenv` field in `conf/config.yaml` are mandatory.


## List of uploaded Space Envs

List here the `base_name` of the uploaded `.yaml`. Include the specific config file only in case that the config file and base name is different.

- elect_dempct_college

## List of Raw Data Sources

These datasets are used to generate the synthetic data. They are described on the README of the `data/` folder.
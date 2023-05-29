import json
import os
import shutil
import yaml
from zipfile import ZipFile
from omegaconf import DictConfig

import hydra

from utils import upload_dataverse_data


def double_zip_folder(folder_path, output_path):
    # Create a temporary zip file
    shutil.make_archive(output_path, "zip", folder_path)

    # Zip the temporary zip file
    zipzip_path = output_path + ".zip.zip"
    with ZipFile(zipzip_path, "w") as f:
        f.write(output_path + ".zip")

    # Remove the temporary zip file
    os.remove(output_path + ".zip")

    return zipzip_path


@hydra.main(config_path="conf", config_name="upload")
def main(cfg: DictConfig):
    # load metadata
    with open(f"{cfg.train_output_path}/metadata.yaml", "r") as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
    dataset_name = metadata["name"]
    odir = f"uploads/{dataset_name}"  # output dir

    # create folder
    shutil.rmtree(odir, ignore_errors=True)
    os.mkdir(odir)

    # metadata.yaml
    shutil.copy(
        f"{cfg.train_output_path}/metadata.yaml",
        f"{odir}/metadata.yaml",
    )
    # synthetic_data.csv
    shutil.copy(
        f"{cfg.train_output_path}/synthetic_data.csv",
        f"{odir}/synthetic_data.csv",
    )
    # leaderboard.csv
    shutil.copy(
        f"{cfg.train_output_path}/leaderboard.csv",
        f"{odir}/leaderboard.csv",
    )
    # config.yaml
    shutil.copy(
        f"{cfg.train_output_path}/.hydra/config.yaml",
        f"{odir}/config.yaml",
    )
    # graph.graphml
    shutil.copy(
        f"{cfg.train_output_path}/graph.graphml",
        f"{odir}/graph.graphml",
    )

    # compress folder and double zip
    zipfile = double_zip_folder(odir, f"{odir}/{dataset_name}")

    # upload to dataverse
    with open(f"{odir}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    data_description = f"""
        This is a synthetic dataset generated for the space project.\n
        The dataset was generated using the following configuration:\n
        {json.dumps(config)}
        """

    dataverse_token = cfg.dataverse.token
    if cfg.dataverse.token is None:
        dataverse_token = os.environ.get("DATAVERSE_TOKEN", None)

    if not cfg.debug:
        if dataverse_token is None:
            raise ValueError(
                "No token provided and DATAVERSE_TOKEN not found in enviroment."
            )

        upload_dataverse_data(
            zipfile,
            data_description,
            cfg.dataverse.baseurl,
            cfg.dataverse.pid,
            dataverse_token,
        )


if __name__ == "__main__":
    main()

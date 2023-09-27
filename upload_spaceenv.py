import json
import logging
import os
import shutil

import hydra
import yaml
from omegaconf import DictConfig
from pyDataverse.api import NativeApi
from pyDataverse.models import Datafile

from utils import double_zip_folder


def upload_dataverse_data(
    data_path: str,
    data_description: str,
    dataverse_baseurl: str,
    dataverse_pid: str,
    token: str,
    publish: bool = False,
):
    """
    Upload data to the collection
    Args:
        file_path (str): Filename
        description (str): Data file description.
        token (str): Dataverse API Token.
    """
    status = "Failed"

    api = NativeApi(dataverse_baseurl, token)

    filename = os.path.basename(data_path)

    dataset = api.get_dataset(dataverse_pid)
    logging.info("Dataverse APIs created.")

    files_list = dataset.json()["data"]["latestVersion"]["files"]
    file2id = {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files_list}
    filename_ = filename.replace(".zip.zip", ".zip")

    if filename_ not in file2id:  # new file
        logging.info("File does not exist in selected dataverse. Creating it.")
        dataverse_datafile = Datafile()
        dataverse_datafile.set(
            {
                "pid": dataverse_pid,
                "filename": filename,
                "description": data_description,
            }
        )
        logging.info("File basename: " + filename)

        resp = api.upload_datafile(dataverse_pid, data_path, dataverse_datafile.json())
        if resp.json()["status"] == "OK":
            logging.info("Dataset uploaded.")
            status = "OK"
        else:
            logging.error("Dataset not uploaded.")
            logging.error(resp.json())

    else:
        logging.info("File already exists. Replacing it.")

        file_id = file2id[filename_]
        json_dict = {
            "description": data_description,
            "forceReplace": True,
            "filename": filename,
        }
        json_str = json.dumps(json_dict)
        resp = api.replace_datafile(file_id, data_path, json_str, is_filepid=False)
        if resp.json()["status"] == "ERROR":
            logging.error(f"An error at replacing the file: {resp.content}")
        else:
            logging.info("Dataset replaced.")
            status = "OK"

    if publish:
        resp = api.publish_dataset(dataverse_pid, release_type="major")
        if resp.json()["status"] == "OK":
            logging.info("Dataset published.")

    return status


@hydra.main(config_path="conf", config_name="upload_spaceenv", version_base=None)
def main(cfg: DictConfig):
    # verify inputs
    assert not cfg.upload or cfg.token is not None, "Token must be provided for upload."

    # load metadata
    train_dir = f"{hydra.utils.get_original_cwd()}/trained_spaceenvs/{cfg.base_name}"

    contents_dir = cfg.base_name
    assert os.path.exists(train_dir), f"Train directory {train_dir} not found."

    if os.path.exists(contents_dir):
        logging.info(f"Target directory {contents_dir} already exists.")
    else:
        os.mkdir(contents_dir)

    # copy relevant files to working directory
    shutil.copy(f"{train_dir}/metadata.yaml", contents_dir)
    # get ext from filename "{train_dir}/synthetic_data*"
    files = os.listdir(train_dir)
    ext = None
    for f in files:
        if "synthetic_data" in f:
            ext = ".".join(f.split(".")[1:])
            break
    if ext is None:
        raise ValueError("Synthetic data not found.")
    shutil.copy(f"{train_dir}/synthetic_data.{ext}", contents_dir)
    shutil.copy(f"{train_dir}/leaderboard.csv", contents_dir)
    shutil.copy(f"{train_dir}/counterfactuals.png", contents_dir)
    shutil.copy(f"{train_dir}/.hydra/config.yaml", f"{contents_dir}/config.yaml")
    # get ext from filename "{train_dir}/graph*"
    ext = None
    for f in files:
        if "graph" in f:
            ext = ".".join(f.split(".")[1:])
            break
    if ext is None:
        raise ValueError("Graph not found.")
    shutil.copy(f"{train_dir}/graph.{ext}", contents_dir)

    # compress folder and double zip
    zipfile = double_zip_folder(contents_dir, f"{cfg.base_name}")

    # upload to dataverse
    with open(f"{contents_dir}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    data_description = f"""
        This is a synthetic dataset generated for the space project.\n
        The dataset was generated using the following configuration:\n
        {json.dumps(config)}
        """

    if cfg.upload:
        status = upload_dataverse_data(
            zipfile,
            data_description,
            cfg.dataverse.baseurl,
            cfg.dataverse.pid,
            token=cfg.token,
            publish=cfg.publish,
        )
    else:
        status = "Upload disabled"

    # save a text file with the upload status, used in the pipeline
    logging.info(f"Upload status: {status}")
    with open("upload_status.txt", "w") as f:
        f.write(status)


if __name__ == "__main__":
    main()

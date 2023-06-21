import json
import os
import shutil
import yaml
from omegaconf import DictConfig
import logging
import hydra

from utils import upload_dataverse_data, double_zip_folder

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="upload", version_base=None)
def main(cfg: DictConfig):
    # load metadata
    train_dir = f"{hydra.utils.get_original_cwd()}/outputs/{cfg.base_name}"
    contents_dir = cfg.base_name
    assert os.path.exists(train_dir), f"Train directory {train_dir} not found."

    if os.path.exists(contents_dir):
        LOGGER.info(f"Target directory {contents_dir} already exists.")
    else:
        os.mkdir(contents_dir)

    # copy relevant files to working directory
    shutil.copy(f"{train_dir}/metadata.yaml", contents_dir)
    shutil.copy(f"{train_dir}/synthetic_data.csv", contents_dir)
    shutil.copy(f"{train_dir}/leaderboard.csv", contents_dir)
    shutil.copy(f"{train_dir}/.hydra/config.yaml", f"{contents_dir}/config.yaml")
    shutil.copy(f"{train_dir}/graph.graphml", contents_dir)

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

    dataverse_token = cfg.dataverse.token
    if cfg.dataverse.token is None:
        dataverse_token = os.environ.get("DATAVERSE_TOKEN", None)

    if dataverse_token is None:
        if not cfg.debug:
            raise ValueError(
                "No token provided and DATAVERSE_TOKEN not found in enviroment."
            )
        else:
            LOGGER.info("No token provided and debug=true. Skipping upload.")
    else:
        upload_dataverse_data(
            zipfile,
            data_description,
            cfg.dataverse.baseurl,
            cfg.dataverse.pid,
            dataverse_token,
            debug=cfg.debug,
            dataset_publish=cfg.publish,
        )


if __name__ == "__main__":
    main()

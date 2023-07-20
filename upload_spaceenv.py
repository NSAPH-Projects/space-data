import json
import os
import shutil
import yaml
from omegaconf import DictConfig
import logging
import hydra

from utils import upload_dataverse_data, double_zip_folder

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="upload_spaceenv", version_base=None)
def main(cfg: DictConfig):
    # verify inputs
    assert not cfg.upload or cfg.token is not None, "Token must be provided for upload."

    # load metadata
    train_dir = f"{hydra.utils.get_original_cwd()}/trained_spaceenvs/{cfg.base_name}"

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

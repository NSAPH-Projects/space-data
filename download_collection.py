import logging
import os

import hydra
from omegaconf import DictConfig
from pyDataverse.api import DataAccessApi, NativeApi


@hydra.main(config_path="conf", config_name="download_collection", version_base=None)
def main(cfg: DictConfig):    
    # connect to dataverse
    api = NativeApi(cfg.dataverse.baseurl)
    data_api = DataAccessApi(cfg.dataverse.baseurl)
    dataset = api.get_dataset(cfg.dataverse.pid)
    files_list = dataset.json()["data"]["latestVersion"]["files"]
    file2id = {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files_list}

    for obj, file in cfg.collection.items():
        if obj == "base_name" or file is None:
            continue
        ext = file.split(".")[-1]
        if ext == "gz":
            ext = file.split(".")[-2] + "." + ext
        path = f"{obj}.{ext}"  # hydra already changed the working directory
        logging.info(f"Downloading {file}.")

        if file not in file2id:
            logging.error(f"File {file} not found in dataverse.")
            raise ValueError(f"File {file} not found in dataverse.")
        else:
            response = data_api.get_datafile(file2id[file])
            with open(path, "wb") as f:
                f.write(response.content)
            logging.info(f"Saved {file} to {path}.")


if __name__ == "__main__":
    main()

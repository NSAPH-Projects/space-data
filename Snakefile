from omegaconf import OmegaConf
import download_collection


# == Load configs ==
if len(config) == 0:
    raise Exception(
        "No config file passed to snakemake."
        " Use flag --configfile conf/pipeline.yaml"
    )

# create OmegaConf from dictionary config
pipeline_cfg = OmegaConf.create(config)  # passed by snakemake when using --configfile
download_cfg = OmegaConf.load("conf/download_collection.yaml")
train_cfg = OmegaConf.load("conf/train_spaceenv.yaml")
upload_cfg = OmegaConf.load("conf/upload_spaceenv.yaml")


# == Collect spaceenv parents ==
spaceenv_parent = {}
for e in pipeline_cfg.spaceenvs:
    cfg = OmegaConf.load(f"conf/spaceenv/{e}.yaml")
    spaceenv_parent[e] = cfg.collection


# == Collect all spaceenvs that required uploading ==
target_files = []
for e in pipeline_cfg.spaceenvs:
    status_file = f"uploads/{e}/upload_status.txt"
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = f.read()
        if status != "OK":
            tgts = [status_file, f"uploads/{e}/{e}.zip.zip"]
            os.remove(status_file)
    else:
        tgts = [status_file, f"uploads/{e}/{e}.zip.zip"]
    target_files.extend([status_file, f"uploads/{e}/{e}.zip.zip"])


def collection_files(c):
    dir = f"data_collections/{c}"
    return [f"{dir}/data.tab", f"{dir}/graph.graphml"]


# == Define rules ==
rule all:
    input:
        target_files,


rule download_data_collection:
    output:
        "data_collections/{collection}/data.tab",
        "data_collections/{collection}/graph.graphml",
    log:
        err="data_collections/{collection}/download_collection.log",
    shell:
        """
        python download_collection.py collection={wildcards.collection} 2> {log.err}
        """


rule train_spaceenv:
    input:
        lambda wildcards: collection_files(spaceenv_parent[wildcards.spaceenv]),
    output:
        "trained_spaceenvs/{spaceenv}/graph.graphml",
        "trained_spaceenvs/{spaceenv}/metadata.yaml",
        "trained_spaceenvs/{spaceenv}/synthetic_data.csv",
        "trained_spaceenvs/{spaceenv}/leaderboard.csv",
    log:
        err="trained_spaceenvs/{spaceenv}/error.log",
    shell:
        """
        python train_spaceenv.py spaceenv={wildcards.spaceenv} 2> {log.err}
        """


rule upload_spaceenv:
    input:
        "trained_spaceenvs/{spaceenv}/graph.graphml",
        "trained_spaceenvs/{spaceenv}/metadata.yaml",
        "trained_spaceenvs/{spaceenv}/synthetic_data.csv",
        "trained_spaceenvs/{spaceenv}/leaderboard.csv",
    params:
        upload=pipeline_cfg.upload,
        token=pipeline_cfg.token,
    log:
        err="uploads/{spaceenv}/upload_spaceenv.log",
    output:
        "uploads/{spaceenv}/{spaceenv}.zip.zip",
        touch("uploads/{spaceenv}/upload_status.txt"),
    shell:
        """
        python upload_spaceenv.py spaceenv={wildcards.spaceenv} \
            upload={params.upload} \
            token={params.token} \
            2> {log.err}
        """

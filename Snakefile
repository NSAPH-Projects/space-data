from omegaconf import OmegaConf


singularity: "docker://mauriciogtec/spacedata:multi"


conda: "requirements.yaml"


# == Load configs ==
if len(config) == 0:
    raise Exception(
        "No config file passed to snakemake."
        " Use flag --configfile conf/pipeline.yaml"
    )

# load configs
pipeline_cfg = OmegaConf.create(config)  # passed by snakemake when using --configfile
download_cfg = OmegaConf.load("conf/download_collection.yaml")
train_cfg = OmegaConf.load("conf/train_spaceenv.yaml")
upload_cfg = OmegaConf.load("conf/upload_spaceenv.yaml")


# == Collect the collections used by each spaceenv ==
spaceenv_parent = {}
for e in pipeline_cfg.spaceenvs:
    cfg = OmegaConf.load(f"conf/spaceenv/{e}.yaml")
    spaceenv_parent[e] = cfg.collection


# == Identify all spaceenvs that required uploading ==
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
    target_files.append(status_file)


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
    params:
        dataverse=pipeline_cfg.download_dataverse,
    shell:
        """
        python download_collection.py \
            collection={wildcards.collection} \
            dataverse={params.dataverse} \
            2> {log.err}
        """


def spaceenv_inputs(wildcards):
    collection = spaceenv_parent[wildcards.spaceenv]
    return [
        f"data_collections/{collection}/data.tab",
        f"data_collections/{collection}/graph.graphml",
    ]


rule train_spaceenv:
    input:
        spaceenv_inputs,
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
        dataverse=pipeline_cfg.upload_dataverse,
    log:
        err="uploads/{spaceenv}/upload_spaceenv.log",
    output:
        "uploads/{spaceenv}/upload_status.txt",
    shell:
        """
        python upload_spaceenv.py spaceenv={wildcards.spaceenv} \
            upload={params.upload} \
            dataverse={params.dataverse} \
            2> {log.err}
        """

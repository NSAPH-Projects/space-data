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

data_collection_data_file_ext = {}
unique_collections = set(spaceenv_parent.values())
for c in unique_collections:
    cfg = OmegaConf.load(f"conf/collection/{c}.yaml")
    data_collection_data_file_ext[c] = cfg.data.split(".")[-1]


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
        "data_collections/{collection}/data.{ext}",
    log:
        err="data_collections/{collection}/download_collection-{ext}.err",
    params:
        dataverse=pipeline_cfg.download_dataverse,
    resources:
        mem_mb=20000,
        disk_mb=20000
    shell:
        """
        python download_collection.py \
            collection={wildcards.collection} \
            dataverse={params.dataverse} \
            2> {log.err}
        """


def train_spaceenv_inputs(wildcards):
    collection = spaceenv_parent[wildcards.spaceenv]
    ext = data_collection_data_file_ext[collection]
    return f"data_collections/{collection}/data.{ext}"


rule train_spaceenv:
    input:
        train_spaceenv_inputs,
    output:
        "trained_spaceenvs/{spaceenv}/synthetic_data.{ext}"
    threads: pipeline_cfg.training_threads
    log:
        err="trained_spaceenvs/{spaceenv}/error-{ext}.log",
    resources:
        mem_mb=60000,
        disk_mb=60000
    shell:
        """
        python train_spaceenv.py spaceenv={wildcards.spaceenv} 2> {log.err}
        """

def upload_spaceenv_inputs(wildcards):
    collection = spaceenv_parent[wildcards.spaceenv]
    ext = data_collection_data_file_ext[collection]
    return "trained_spaceenvs/{spaceenv}/synthetic_data." + ext


rule upload_spaceenv:
    input:
        upload_spaceenv_inputs,
    params:
        upload=pipeline_cfg.upload,
        dataverse=pipeline_cfg.upload_dataverse,
        token=pipeline_cfg.token,
    log:
        err="uploads/{spaceenv}/upload_spaceenv.err",
    output:
        "uploads/{spaceenv}/upload_status.txt",
    resources:
        mem_mb=10000,
        disk_mb=10000
    shell:
        """
        python upload_spaceenv.py spaceenv={wildcards.spaceenv} \
            upload={params.upload} \
            dataverse={params.dataverse} \
            token={params.token} \
            2> {log.err}
        """

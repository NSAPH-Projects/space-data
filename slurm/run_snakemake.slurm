#!/bin/bash

#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 0-12:00:00
#SBATCH -p shared
#SBATCH --mem=88GB
#SBATCH -o slurm/slurm.%N.%j.out
#SBATCH -e slurm/slurm.%N.%j.err
#SBATCH --mail-type=ALL

# Params
dataverse="harvard"            # harvard or demo
singularity_within_snakemake=1 # 1 or 0
upload=0                       # 1 or 0

# Set token to DEMO_DATAVERSE_TOKEN or HARVARD_DATAVERSE_TOKEN
if [ $dataverse == "harvard" ]; then
    token=$HARVARD_DATAVERSE_TOKEN
elif [ $dataverse == "demo" ]; then
    token=$DEMO_DATAVERSE_TOKEN
else
    echo "Dataverse not recognized. Please use harvard or demo."
    exit 1
fi

if [ $singularity_within_snakemake == 1 ]; then
    job="snakemake --rerun-incomplete --nolock -j --use-singularity"
else
    singularity_file="spacedata_multi.sif"
    if [ ! -f $singularity_file ]; then
        singularity pull --name docker://mauriciogtec/spacedata:multi
    fi
    job="singularity exec $singularity_file snakemake --rerun-incomplete --nolock -j"
fi

options="--configfile conf/pipeline.yaml -C \
    upload=$upload token=$token upload_dataverse=$dataverse"

$job $options

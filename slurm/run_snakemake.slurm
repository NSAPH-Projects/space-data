#!/bin/bash

#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 0-24:00:00
#SBATCH -p shared
#SBATCH --mem=64GB
#SBATCH -o slurm/logs/slurm.%N.%j.out
#SBATCH -e slurm/logs/slurm.%N.%j.err
#SBATCH --mail-type=ALL

singularity_file=spacedata_multi.sif
dataverse_api=$DEMO_DATAVERSE_TOKEN
job="snakemake -j --configfile conf/pipeline.yaml"
options="-C upload=true token=$dataverse_api upload_dataverse=demo"

# Pull from docker hub if singularity image (.sift) doesn't exist.
# Normally, it would be sufficient to run snakemake with --use-singularity
# but a bug in Snakemake prevents it. The bug has been identified and fixed
# for the next release of Snakemake (v7.30.2).
if [ ! -f $singularity_file ]; then
    singularity pull --name docker://mauriciogtec/spacedata:multi
fi

singularity exec $singularity_file $job $options

#!/bin/bash

#SBATCH -n 1
#SBATCH -t 0-24:00:00
#SBATCH -p shared
#SBATCH --mem=8G
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

job="snakemake --rerun-incomplete --nolock --use-singularity"
slurm_options="/usr/bin/sbatch --ntasks {cluster.ntasks} -N {cluster.N} -t {cluster.t} \
    --cpus-per-task {cluster.cpus_per_task} -p {cluster.p} --mem {cluster.mem} -o {cluster.output} \
    -e {cluster.error} --mail-type={cluster.mail_type}"
options="--configfile conf/pipeline.yaml -C upload=$upload token=$token upload_dataverse=$dataverse"

$job $options --cluster "${slurm_options}" --cluster-config conf/cluster.yaml -j 12
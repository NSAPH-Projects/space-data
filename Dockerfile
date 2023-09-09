# dockerfile from miniconda image copy local requirements.yaml and install
FROM continuumio/miniconda3:23.3.1-0

# set working directory
WORKDIR /app

# install gcc and python3-dev
RUN apt-get update && apt-get install -y gcc python3-dev

# install requirements
RUN conda install -c conda-forge mamba
COPY requirements.yaml requirements.yaml
RUN mamba env update -n base -f requirements.yaml

# there's currently a bug with ray on linux
RUN pip uninstall -y ray

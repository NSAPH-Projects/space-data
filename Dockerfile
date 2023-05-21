# dockerfile from miniconda image copy local requirements.yaml and install
FROM continuumio/miniconda3:23.3.1-0

# set working directory
WORKDIR /app

# install gcc and python3-dev
RUN apt-get update && apt-get install -y gcc python3-dev

# copy the requirements file
COPY requirements.yaml requirements.yaml

# install dependencies
RUN conda env create -f requirements.yaml

# maek default conda env
RUN echo "export CONDA_DEFAULT_ENV=spacedata" >> ~/.bashrc
RUN echo "export PATH=/opt/conda/envs/spacedata/bin:\$PATH" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "spacedata", "/bin/bash", "-c"]

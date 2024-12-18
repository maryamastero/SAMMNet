#!/bin/bash

conda create -n deepenv python=3.9 -y

source $(conda info --base)/etc/profile.d/conda.sh

conda activate deepenv

conda install pytorch torchvision torchaudio -c pytorch -c conda-forge -y

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

conda install -c conda-forge rdkit -y

pip install mlflow

pip install streamlit

pip install fastai

conda install -c anaconda ipykernel

echo "Setup complete. To activate the environment, use 'conda activate deepenv'."

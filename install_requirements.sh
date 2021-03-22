#!/bin/bash

venv_dir=./venv

if [ ! -d $venv_dir ] 
then
    echo "Creating a conda environment in " $venv_dir
    conda create -y -p $venv_dir
    source activate $venv_dir
    echo "Install the required C++ libraries..."
    conda install -y -c conda-forge blas doxygen xtensor xtensor-blas xtensor-python
fi


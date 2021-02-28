#!/bin/bash

build_dir=../../build/bindings/python
venv_dir=../../venv

if [ ! -d $build_dir ] 
then
    mkdir -p $build_dir    
fi

if [ ! -d $venv_dir ] 
then
    echo "Creating a conda environment in " $venv_dir
    conda create -p $venv_dir python=3.8
fi

echo "Using conda environment in " $venv_dir
activate $venv_dir
pip install -r requirements.txt

echo "Create the Python package in " $build_dir
python3 setup.py bdist_wheel --dist-dir=$build_dir

echo "Clean up temporary folders..."
rm -r -f ./ado.egg-info
rm -r -f ./build
rm -r -f ./tmp

echo "Done !"
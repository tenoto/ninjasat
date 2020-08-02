#!/bin/bash 

echo "####################"
echo "#     NinjaSat     #"
echo "####################"

export NINJASAT_PATH=$(pwd)
export PYTHONPATH=$NINJASAT_PATH:$PYTHONPATH

export PATH=$NINJASAT_PATH/cubesat/cli:$PATH




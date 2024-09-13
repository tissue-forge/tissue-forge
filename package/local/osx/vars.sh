#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
source ${this_dir}/site_vars.sh

if [ ! -d "${TFPYSITEDIR}" ]; then 
    echo "*TF* site-packages not found (TFPYSITEDIR=${TFPYSITEDIR})"
    exit 1
fi
if [ ! -d "${TFENV}" ]; then 
    echo "*TF* Environment not found (TFENV=${TFENV})"
    exit 1
fi

export PYTHONPATH=${TFPYSITEDIR}:${PYTHONPATH}

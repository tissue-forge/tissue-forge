#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
source ${this_dir}/site_vars.sh

if [[ ! -d "${TFPYSITEDIR}" ]]; then 
    exit 1
fi
if [[ ! -d "${TFENV}" ]]; then 
    exit 2
fi

export PYTHONPATH=${TFPYSITEDIR}:${PYTHONPATH}

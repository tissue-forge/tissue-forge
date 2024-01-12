#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
TFENV=${this_dir}/env

if [[ ! -d "${TFCONDAENV}" ]]; then
    TFCONDAENV=${HOME}/miniconda3/etc/profile.d/conda.sh
fi

source ${TFCONDAENV}
conda create --yes --prefix ${TFENV}
conda env update --prefix ${TFENV} --file ${this_dir}/rtenv.yml
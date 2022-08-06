#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
TFENV=${this_dir}/env

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda create --yes --prefix ${TFENV}
conda env update --prefix ${TFENV} --file ${this_dir}/rtenv.yml
#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda create --yes --prefix ${TFENV}
conda env update --prefix ${TFENV} --file ${TFSRCDIR}/package/local/linux/env.yml

#!/bin/bash

source ${TFCONDAENV}
conda create --yes --prefix ${TFENV}
conda env update --prefix ${TFENV} --file ${TFSRCDIR}/package/local/osx/env.yml

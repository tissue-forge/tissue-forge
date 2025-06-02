#!/bin/bash

if [ -n "${TFENVNEEDSCONDA+x}" ]; then
    echo "*TF* Using conda at ${TFCONDAENV}"

    source ${TFCONDAENV}
    if [ $? -ne 0 ]; then
        echo "*TF* Something went wrong when sourcing conda ($?)."
        echo "*TF* If conda is already available in this environment, then this check can be disabled by not setting TFENVNEEDSCONDA"
        exit $?
    fi
fi

conda create --yes --prefix ${TFENV}
conda env update --prefix ${TFENV} --file ${TFSRCDIR}/package/local/linux/env.yml

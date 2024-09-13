#!/bin/bash

if [ ! -z "${TFENVNEEDSCONDA+x}" ]; then
    echo "*TF* Using conda at ${TFCONDAENV}"

    source ${TFCONDAENV}
    if [ $? -ne 0 ]; then 
        echo "*TF* Something went wrong when sourcing conda ($?)."
        echo "*TF* If conda is already available in this environment, then this check can be disabled by not setting TFENVNEEDSCONDA"
        exit $?
    fi
fi

conda create --yes --prefix ${TFENV}
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong when creating the environment ($?)."
    exit $?
fi

conda env update --prefix ${TFENV} --file ${TFSRCDIR}/package/local/linux/env.yml
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong when populating the environment ($?)."
    exit $?
fi

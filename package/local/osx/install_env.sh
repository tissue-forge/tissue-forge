#!/bin/bash

echo "*TF* Using conda at ${TFCONDAENV}"

source ${TFCONDAENV}
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong when sourcing conda ($?)."
    exit $?
fi

conda create --yes --prefix ${TFENV}
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong when creating the environment ($?)."
    exit $?
fi

conda env update --prefix ${TFENV} --file ${TFSRCDIR}/package/local/osx/env.yml
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong when populating the environment ($?)."
    exit $?
fi

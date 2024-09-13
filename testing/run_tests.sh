#!/bin/bash

if [[ $(uname) == Darwin ]]; then
    echo "*TF* *************************************"
    echo "*TF* Launching Tissue Forge tests for OSX"
    echo "*TF* *************************************"
else
    echo "*TF* **************************************"
    echo "*TF* Launching Tissue Forge tests for Linux"
    echo "*TF* **************************************"
fi

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

TFTESTS_TESTSDIR=${this_dir}/build
if [ ! -d "${TFENV}" ]; then
    echo "*TF* Environment not found (TFENV=${TFENV})"
    exit 1
fi
if [[ $(uname) != Darwin ]]; then
    export LD_LIBRARY_PATH=${TFENV}/lib:${LD_LIBRARY_PATH}
fi
cd ${TFTESTS_TESTSDIR}
ctest --output-on-failure

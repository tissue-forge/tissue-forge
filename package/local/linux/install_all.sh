#!/bin/bash

if [ ! -d "${TFSRCDIR}" ]; then 
    echo "*TF* Source directory not found (TFSRCDIR=${TFSRCDIR})"
    exit 1
fi

bash ${TFSRCDIR}/package/local/linux/install_core.sh
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with installing core ($?)."
    exit $?
fi

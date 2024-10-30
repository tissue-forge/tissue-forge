#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

current_dir=$(pwd)

set -o pipefail -e

if [[ $(uname) == Darwin ]]; then
    echo "*TF* *******************************************"
    echo "*TF* Launching Tissue Forge local build for OSX"
    echo "*TF* *******************************************"

    subdir=osx
else
    echo "*TF* ********************************************"
    echo "*TF* Launching Tissue Forge local build for Linux"
    echo "*TF* ********************************************"

    subdir=linux
fi

source ${this_dir}/${subdir}/install_vars.sh

if [ ! -d "${TFSRCDIR}" ]; then 
    echo "*TF* Source directory not found (TFSRCDIR=${TFSRCDIR})"
    exit 1
fi

bash ${TFSRCDIR}/package/local/${subdir}/install_env.sh
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with installing the environment ($?)."
    exit $?
fi

if [ ! -z "${TFENVNEEDSCONDA+x}" ]; then
    source ${TFCONDAENV}
fi

# Install CUDA support if requested
if [ -z "${TF_WITHCUDA+x}" ]; then
    if [ ${TF_WITHCUDA} -eq 1 ]; then 
        # Validate specified compute capability
        if [ ! -z "${CUDAARCHS+x}" ]; then
            echo "*TF* No compute capability specified"
            exit 1
        fi

        echo "*TF* Detected CUDA support request"
        echo "*TF* Installing additional dependencies..."

        export TFCUDAENV=${TFENV}
        conda install -y -c nvidia -p ${TFENV} cuda
        if [ $? -ne 0 ]; then 
            echo "*TF* Something went wrong with installing CUDA ($?)."
            exit $?
        fi
    fi
fi

conda activate ${TFENV}

bash ${TFSRCDIR}/package/local/${subdir}/install_all.sh
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with installing Tissue Forge ($?)."
    exit $?
fi

cd ${current_dir}

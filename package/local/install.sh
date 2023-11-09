#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

current_dir=$(pwd)

set -o pipefail -e

if [[ $(uname) == Darwin ]]; then
    subdir=osx
else
    subdir=linux
fi

source ${this_dir}/${subdir}/install_vars.sh

bash ${TFSRCDIR}/package/local/${subdir}/install_env.sh

source ${TFCONDAENV}

# Install CUDA support if requested
if [ -z "${TF_WITHCUDA+x}" ]; then
    if [ ${TF_WITHCUDA} -eq 1 ]; then 
        # Validate specified compute capability
        if [ ! -z "${CUDAARCHS+x}" ]; then
            echo "No compute capability specified"
            exit 1
        fi

        echo "Detected CUDA support request"
        echo "Installing additional dependencies..."

        export TFCUDAENV=${TFENV}
        conda install -y -c nvidia -p ${TFENV} cuda
    fi
fi

conda activate ${TFENV}

bash ${TFSRCDIR}/package/local/${subdir}/install_all.sh

cd ${current_dir}

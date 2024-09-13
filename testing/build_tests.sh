#!/bin/bash

if [[ $(uname) == Darwin ]]; then
    echo "*TF* *******************************************"
    echo "*TF* Launching Tissue Forge tests build for OSX"
    echo "*TF* *******************************************"
else
    echo "*TF* ********************************************"
    echo "*TF* Launching Tissue Forge tests build for Linux"
    echo "*TF* ********************************************"
fi

current_dir=$(pwd)

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

if [ ! -d "${TFENV}" ]; then
    echo "*TF* Environment not found (TFENV=${TFENV})"
    exit 1
elif [ ! -d "${TFINSTALLDIR}" ]; then
    echo "*TF* Installation not found (TFINSTALLDIR=${TFINSTALLDIR})"
    exit 1
fi

TFTESTS_BUILDDIR=${this_dir}/build

mkdir ${TFTESTS_BUILDDIR}

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH="${TFENV};${TFINSTALLDIR};${TFINSTALLDIR}/lib")
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${TFENV}/bin/python)

if [[ $(uname) == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=${TFOSX_SYSROOT}
else
    CMAKE_CONFIG_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)
fi

cd ${this_dir}

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      -S . \
      -B "${TFTESTS_BUILDDIR}"
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with configuring the build ($?)."
    exit $?
fi

cmake --build "${TFTESTS_BUILDDIR}" --config ${TFBUILD_CONFIG}
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with the build ($?)."
    exit $?
fi

cd ${current_dir}

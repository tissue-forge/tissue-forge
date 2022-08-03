#!/bin/bash

current_dir=$(pwd)

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

if [[ ! -d "${TFENV}" ]]; then
    exit 1
elif [[ ! -d "${TFINSTALLDIR}" ]]; then
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

    export CC=${TFENV}/bin/clang
    export CXX=${TFENV}/bin/clang++
fi

cd ${this_dir}

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      -S . \
      -B "${TFTESTS_BUILDDIR}"

cmake --build "${TFTESTS_BUILDDIR}" --config ${TFBUILD_CONFIG}

cd ${current_dir}

#!/bin/bash

if [[ ! -d "${TFENV}" ]]; then
    exit 1
elif [[ ! -d "${TFSRCDIR}" ]]; then
    exit 2
fi

current_dir=$(pwd)

mkdir -p -v ${TFBUILDDIR}
mkdir -p -v ${TFINSTALLDIR}

cd ${TFBUILDDIR}

export MACOSX_DEPLOYMENT_TARGET=${TFOSX_SYSROOT}
export CONDA_BUILD_SYSROOT="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX${TFOSX_SYSROOT}.sdk"

if [[ ! -d "${CONDA_BUILD_SYSROOT}" ]]; then
    export CONDA_BUILD_SYSROOT="$(xcrun --sdk macosx${TFOSX_SYSROOT} --show-sdk-path)"
fi

if [[ ! -d "${CONDA_BUILD_SYSROOT}" ]]; then
    echo "SDK not found"
    exit 3
fi

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS+=(-DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${TFENV}/bin/python)
CMAKE_CONFIG_ARGS+=(-DLIBXML_INCLUDE_DIR:PATH=${TFENV}/include/libxml2)

if [[ $(uname -m) == 'arm64' ]]; then
    CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_AVX:BOOL=OFF)
    CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_SSE4:BOOL=OFF)
fi

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${TFSRCDIR}"

cmake --build . --config ${TFBUILD_CONFIG} --target install

cd ${current_dir}

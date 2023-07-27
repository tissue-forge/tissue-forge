#!/bin/bash

if [[ ! -d "${TFENV}" ]]; then
    exit 1
elif [[ ! -d "${TFSRCDIR}" ]]; then
    exit 2
fi

current_dir=$(pwd)

mkdir -p -v ${TFBUILDDIR}
mkdir -p -v ${TFBUILDDIR}/libroadrunner-deps
mkdir -p -v ${TFBUILDDIR}/roadrunner
mkdir -p -v ${TFBUILDDIR}/antimony
mkdir -p -v ${TFBUILDDIR}/tissue-forge
mkdir -p -v ${TFINSTALLDIR}

export MACOSX_DEPLOYMENT_TARGET=${TFOSX_SYSROOT}
export CONDA_BUILD_SYSROOT="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX${TFOSX_SYSROOT}.sdk"

if [[ ! -d "${CONDA_BUILD_SYSROOT}" ]]; then
    export CONDA_BUILD_SYSROOT="$(xcrun --sdk macosx${TFOSX_SYSROOT} --show-sdk-path)"
fi

if [[ ! -d "${CONDA_BUILD_SYSROOT}" ]]; then
    echo "SDK not found"
    exit 3
fi

# Build libRoadRunner dependencies

cd ${TFBUILDDIR}/libroadrunner-deps

declare -a CMAKE_CONFIG_ARGS_RRD

CMAKE_CONFIG_ARGS_RRD+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS_RRD+=(-DCMAKE_PREFIX_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS_RRD+=(-DCMAKE_FIND_ROOT_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS_RRD+=(-DCMAKE_INSTALL_PREFIX:PATH=${TFBUILDDIR}/libroadrunner-deps/install)
CMAKE_CONFIG_ARGS_RRD+=(-DUSE_UNIVERSAL_BINARIES:BOOL=OFF)
CMAKE_CONFIG_ARGS_RRD+=(-DWITH_ZLIB:BOOL=OFF)
CMAKE_CONFIG_ARGS_RRD+=(-DCMAKE_OSX_ARCHITECTURES:STRING=$(uname -m))

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS_RRD[@]}" \
      "${TFSRCDIR}/extern/libroadrunner-deps"

cmake --build . --config ${TFBUILD_CONFIG} --target install

# Build libRoadRunner

cd ${TFBUILDDIR}/roadrunner

declare -a CMAKE_CONFIG_ARGS_RR

CMAKE_CONFIG_ARGS_RR+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS_RR+=(-DCMAKE_PREFIX_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS_RR+=(-DCMAKE_FIND_ROOT_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS_RR+=(-DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR}/lib/TissueForge/roadrunner)
CMAKE_CONFIG_ARGS_RR+=(-DRR_DEPENDENCIES_INSTALL_PREFIX:PATH=${TFBUILDDIR}/libroadrunner-deps/install)
CMAKE_CONFIG_ARGS_RR+=(-DLLVM_INSTALL_PREFIX:PATH=${TFENV})
CMAKE_CONFIG_ARGS_RR+=(-DCMAKE_OSX_ARCHITECTURES:STRING=$(uname -m))

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS_RR[@]}" \
      "${TFSRCDIR}/extern/roadrunner"

cmake --build . --config ${TFBUILD_CONFIG} --target install

# Build libAntimony

cd ${TFBUILDDIR}/antimony

declare -a CMAKE_CONFIG_ARGS_ANT

CMAKE_CONFIG_ARGS_ANT+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS_ANT+=(-DCMAKE_PREFIX_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS_ANT+=(-DCMAKE_FIND_ROOT_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS_ANT+=(-DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR}/lib/TissueForge/antimony)
CMAKE_CONFIG_ARGS_ANT+=(-DLIBSBML_INSTALL_DIR:PATH=${TFINSTALLDIR}/lib/TissueForge/roadrunner)
CMAKE_CONFIG_ARGS_ANT+=(-DUSE_UNIVERSAL_BINARIES:BOOL=OFF)
CMAKE_CONFIG_ARGS_ANT+=(-DWITH_QTANTIMONY:BOOL=OFF)
CMAKE_CONFIG_ARGS_ANT+=(-DCMAKE_OSX_ARCHITECTURES:STRING=$(uname -m))

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS_ANT[@]}" \
      "${TFSRCDIR}/extern/antimony"

cmake --build . --config ${TFBUILD_CONFIG} --target install

# Build Tissue Forge

cd ${TFBUILDDIR}/tissue-forge

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS+=(-DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${TFENV}/bin/python)
CMAKE_CONFIG_ARGS+=(-DLIBXML_INCLUDE_DIR:PATH=${TFENV}/include/libxml2)
CMAKE_CONFIG_ARGS+=(-DCMAKE_OSX_ARCHITECTURES:STRING=$(uname -m))

if [[ $(uname -m) == 'arm64' ]]; then
    CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_AVX:BOOL=OFF)
    CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_SSE4:BOOL=OFF)
fi

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${TFSRCDIR}"

cmake --build . --config ${TFBUILD_CONFIG} --target install

cd ${current_dir}

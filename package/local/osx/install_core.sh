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

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${TFENV})
CMAKE_CONFIG_ARGS+=(-DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${TFENV}/bin/python)
CMAKE_CONFIG_ARGS+=(-DLIBXML_INCLUDE_DIR:PATH=${TFENV}/include/libxml2)

if [ -n "${TF_OSX_DEPLOYMENT_TARGET+x}" ]; then
  echo "*TF* Using externally specified deployment target"

  CMAKE_CONFIG_ARGS+=(-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${TF_OSX_DEPLOYMENT_TARGET})
fi

if [ -n "${TF_BUILD_SYSROOT+x}" ]; then
  echo "*TF* Using externally specified SDK"

  if [ ! -d "${TF_BUILD_SYSROOT}" ]; then
      echo "*TF* SDK not found"

      exit 1
  fi

  CMAKE_CONFIG_ARGS+=(-DCMAKE_OSX_SYSROOT:PATH=${TF_BUILD_SYSROOT})
fi

if [[ $(uname -m) == 'arm64' ]]; then
    echo "*TF* Detected arm64"

    CMAKE_CONFIG_ARGS+=(-DCMAKE_APPLE_SILICON_PROCESSOR:STRING=arm64)
    CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_AVX:BOOL=OFF)
    CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_SSE4:BOOL=OFF)
fi

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${TFSRCDIR}"

cmake --build . --config ${TFBUILD_CONFIG} --target install

cd ${current_dir}

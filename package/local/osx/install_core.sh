#!/bin/bash

echo "*TF* **************** Tissue Forge core build: start ****************"
echo "*TF* Executing Tissue Forge local build with the following parameters "
echo "*TF* TFBUILD_CONFIG ${TFBUILD_CONFIG}"
echo "*TF* TFSRCDIR ${TFSRCDIR}"
echo "*TF* TFBUILDDIR ${TFBUILDDIR}"
echo "*TF* TFINSTALLDIR ${TFINSTALLDIR}"
echo "*TF* TFENV ${TFENV}"
echo "*TF* TFBUILDQUAL ${TFBUILDQUAL}"
echo "*TF* TFENVNEEDSCONDA ${TFENVNEEDSCONDA}"
echo "*TF* TFCONDAENV ${TFCONDAENV}"
echo "*TF* TF_BUILD_SYSROOT ${TF_BUILD_SYSROOT}"
echo "*TF* TF_OSX_DEPLOYMENT_TARGET ${TF_OSX_DEPLOYMENT_TARGET}"
echo "*TF* TFPACKAGELOCALOFF ${TFPACKAGELOCALOFF}"
echo "*TF* TFPACKAGECONDA ${TFPACKAGECONDA}"
echo "*TF* JSON_INCLUDE_DIRS ${JSON_INCLUDE_DIRS}"
echo "*TF* **************************************************************"

if [ ! -d "${TFENV}" ]; then
    echo "*TF* Environment not found (TFENV=${TFENV})"
    exit 1
elif [ ! -d "${TFSRCDIR}" ]; then
    echo "*TF* Source not found (TFSRCDIR=${TFSRCDIR})"
    exit 1
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

echo "*TF* **************************************************************"

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${TFSRCDIR}"
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with configuring the build ($?)."
    exit $?
fi

cmake --build . --config ${TFBUILD_CONFIG} --target install
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with the build ($?)."
    exit $?
fi

cd ${current_dir}

echo "*TF* ***************** Tissue Forge core build: end *****************"

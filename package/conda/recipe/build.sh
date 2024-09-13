#!/bin/bash

if [[ $(uname) == Darwin ]]; then
    echo "*TF* *******************************************"
    echo "*TF* Launching Tissue Forge conda build for OSX"
    echo "*TF* *******************************************"
else
    echo "*TF* ********************************************"
    echo "*TF* Launching Tissue Forge conda build for Linux"
    echo "*TF* ********************************************"
fi

TFBUILD_CONFIG=Release

export TFPACKAGELOCALOFF=1
export TFPACKAGECONDA=1

echo "*TF* Executing Tissue Forge local build with the following parameters "
echo "*TF* TFBUILD_CONFIG ${TFBUILD_CONFIG}"
echo "*TF* TFPACKAGELOCALOFF ${TFPACKAGELOCALOFF}"
echo "*TF* TFPACKAGECONDA ${TFPACKAGECONDA}"
echo "*TF* PREFIX ${PREFIX}"
echo "*TF* SRC_DIR ${SRC_DIR}"
if [[ $(uname) == Darwin ]]; then
  echo "*TF* TFOSX_SYSROOT ${TFOSX_SYSROOT}"
  echo "*TF* CONDA_BUILD_SYSROOT ${CONDA_BUILD_SYSROOT}"
  echo "*TF* TF_OSXARM64 ${TF_OSXARM64}"
else 
  echo "*TF* CONDA_PREFIX ${CONDA_PREFIX}"
fi
echo "*TF* ****************************************************************"

mkdir -p -v tf_build_conda
cd tf_build_conda

declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG})
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DPython_EXECUTABLE:PATH=${PREFIX}/bin/python)
CMAKE_CONFIG_ARGS+=(-DLIBXML_INCLUDE_DIR:PATH=${PREFIX}/include/libxml2)

if [[ $(uname) == Darwin ]]; then
  export MACOSX_DEPLOYMENT_TARGET=${TFOSX_SYSROOT}
  CMAKE_CONFIG_ARGS+=(-DCMAKE_OSX_SYSROOT:PATH=${CONDA_BUILD_SYSROOT})

  if [ -z "${TF_OSXARM64+x}" ]; then
    if [ ${TF_OSXARM64} -eq 1 ]; then
      CMAKE_CONFIG_ARGS+=(-DCMAKE_APPLE_SILICON_PROCESSOR:STRING=arm64)
      CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_AVX:BOOL=OFF)
      CMAKE_CONFIG_ARGS+=(-DTF_ENABLE_SSE4:BOOL=OFF)
    fi
  fi

else
  CMAKE_CONFIG_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)

  # Helping corrade rc find the right libstdc++
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
fi

echo "*TF* ****************************************************************"

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${SRC_DIR}"
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with configuring the build ($?)."
    exit $?
fi

cmake --build . --config ${TFBUILD_CONFIG} --target install
if [ $? -ne 0 ]; then 
    echo "*TF* Something went wrong with the build ($?)."
    exit $?
fi

#!/bin/bash

TFBUILD_CONFIG=Release

export TFPACKAGELOCALOFF=1
export TFPACKAGECONDA=1

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
else
  CMAKE_CONFIG_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)

  export CC=${CONDA_PREFIX}/bin/clang
  export CXX=${CONDA_PREFIX}/bin/clang++

  # Helping corrade rc find the right libstdc++
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
fi

cmake -G "Ninja" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${SRC_DIR}"

cmake --build . --config ${TFBUILD_CONFIG} --target install

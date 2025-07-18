#!/bin/bash

if [[ ! -d "${TFENV}" ]]; then
    exit 1
elif [[ ! -d "${TFSRCDIR}" ]]; then
    exit 2
fi

# Helping corrade rc find the right libstdc++
export LD_LIBRARY_PATH=${TFENV}/lib:${LD_LIBRARY_PATH}

current_dir=$(pwd)

mkdir -p -v ${TFBUILDDIR}
mkdir -p -v ${TFINSTALLDIR}

cd ${TFBUILDDIR}

cmake -DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${TFENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${TFENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR} \
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
      -DPython_EXECUTABLE:PATH=${TFENV}/bin/python \
      -DLIBXML_INCLUDE_DIR:PATH=${TFENV}/include/libxml2 \
      -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=${TFCUDAENV} \
      "${TFSRCDIR}"

cmake --build . --config ${TFBUILD_CONFIG} --target install

cd ${current_dir}

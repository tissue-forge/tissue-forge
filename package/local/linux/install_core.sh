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

# Build libRoadRunner dependencies

cd ${TFBUILDDIR}/libroadrunner-deps

cmake -DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${TFENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${TFENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${TFBUILDDIR}/libroadrunner-deps/install \
      -DWITH_ZLIB:BOOL=OFF \
      "${TFSRCDIR}/extern/libroadrunner-deps"

cmake --build . --config ${TFBUILD_CONFIG} --target install

# Build libRoadRunner

cd ${TFBUILDDIR}/roadrunner

cmake -DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${TFENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${TFENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR}/lib/TissueForge/roadrunner \
      -DRR_DEPENDENCIES_INSTALL_PREFIX:PATH=${TFBUILDDIR}/libroadrunner-deps/install \
      -DLLVM_INSTALL_PREFIX:PATH=${TFENV} \
      "${TFSRCDIR}/extern/roadrunner"

cmake --build . --config ${TFBUILD_CONFIG} --target install

# Build libAntimony

cd ${TFBUILDDIR}/antimony

cmake -DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${TFENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${TFENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR}/lib/TissueForge/antimony \
      -DLIBSBML_INSTALL_DIR:PATH=${TFINSTALLDIR}/lib/TissueForge/roadrunner \
      -DWITH_QTANTIMONY:BOOL=OFF \
      "${TFSRCDIR}/extern/antimony"

cmake --build . --config ${TFBUILD_CONFIG} --target install

# Build Tissue Forge

cd ${TFBUILDDIR}/tissue-forge

cmake -DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${TFENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${TFENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR} \
      -DPython_EXECUTABLE:PATH=${TFENV}/bin/python \
      -DLIBXML_INCLUDE_DIR:PATH=${TFENV}/include/libxml2 \
      -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=${TFCUDAENV} \
      "${TFSRCDIR}"

cmake --build . --config ${TFBUILD_CONFIG} --target install

cd ${current_dir}

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
echo "*TF* TFCUDAENV ${TFCUDAENV}"
echo "*TF* TF_WITHCUDA ${TF_WITHCUDA}"
echo "*TF* CUDAARCHS ${CUDAARCHS}"
echo "*TF* TFPACKAGELOCALOFF ${TFPACKAGELOCALOFF}"
echo "*TF* TFPACKAGECONDA ${TFPACKAGECONDA}"
echo "*TF* JSON_INCLUDE_DIRS ${JSON_INCLUDE_DIRS}"
echo "*TF* **************************************************************"
echo "*TF* Setting up for Tissue Forge local build "

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

echo "*TF* **************************************************************"

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

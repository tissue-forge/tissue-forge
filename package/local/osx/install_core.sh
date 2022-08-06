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
    echo "SDK not found"
    exit 3
fi

cmake -DCMAKE_BUILD_TYPE:STRING=${TFBUILD_CONFIG} \
      -G "Ninja" \
      -DCMAKE_PREFIX_PATH:PATH=${TFENV} \
      -DCMAKE_FIND_ROOT_PATH:PATH=${TFENV} \
      -DCMAKE_INSTALL_PREFIX:PATH=${TFINSTALLDIR} \
      -DPython_EXECUTABLE:PATH=${TFENV}/bin/python \
      -DLIBXML_INCLUDE_DIR:PATH=${TFENV}/include/libxml2 \
      "${TFSRCDIR}"

cmake --build . --config ${TFBUILD_CONFIG} --target install

cd ${current_dir}

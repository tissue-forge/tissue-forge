#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

# build configuration
export TFBUILD_CONFIG=Release

# path to source root
export TFSRCDIR=${this_dir}/../../..

# path to build root
export TFBUILDDIR=${TFSRCDIR}/../tissue-forge_build

# path to install root
export TFINSTALLDIR=${TFSRCDIR}/../tissue-forge_install

# path to environment root
export TFENV=${TFINSTALLDIR}/env

# local build qualifier
export TFBUILDQUAL=local

# path to cuda root directory
export TFCUDAENV=""

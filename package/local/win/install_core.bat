@echo off

echo "*TF* **************** Tissue Forge core build: start ****************"
echo "*TF* Executing Tissue Forge local build with the following parameters "
echo "*TF* TFBUILD_CONFIG %TFBUILD_CONFIG%"
echo "*TF* TFSRCDIR %TFSRCDIR%"
echo "*TF* TFBUILDDIR %TFBUILDDIR%"
echo "*TF* TFINSTALLDIR %TFINSTALLDIR%"
echo "*TF* TFENV %TFENV%"
echo "*TF* TFBUILDQUAL %TFBUILDQUAL%"
echo "*TF* TFCUDAENV %TFCUDAENV%"
echo "*TF* TF_WITHCUDA %TF_WITHCUDA%"
echo "*TF* CUDAARCHS %CUDAARCHS%"
echo "*TF* TFPACKAGELOCALOFF %TFPACKAGELOCALOFF%"
echo "*TF* TFPACKAGECONDA %TFPACKAGECONDA%"
echo "*TF* JSON_INCLUDE_DIRS %JSON_INCLUDE_DIRS%"
echo "*TF* **************************************************************"

if not exist "%TFENV%" (
      echo "*TF* Environment not found (TFENV=%TFENV%)"
      exit /B 1
) else if not exist "%TFSRCDIR%" (
      echo "*TF* Source not found (TFSRCDIR=%TFSRCDIR%)"
      exit /B 1
)

set current_dir=%cd%

mkdir %TFBUILDDIR%
mkdir %TFINSTALLDIR%

cd %TFBUILDDIR%

echo "*TF* **************************************************************"

cmake -DCMAKE_BUILD_TYPE:STRING=%TFBUILD_CONFIG% ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH:PATH=%TFENV% ^
      -DCMAKE_FIND_ROOT_PATH:PATH=%TFENV%\Library ^
      -DCMAKE_INSTALL_PREFIX:PATH=%TFINSTALLDIR% ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPython_EXECUTABLE:PATH=%TFENV%\python.exe ^
      -DPThreads_ROOT:PATH=%TFENV%\Library ^
      -DLIBXML_INCLUDE_DIR:PATH=%TFENV%\Library\include\libxml2 ^
      -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=%TFCUDAENV% ^
      "%TFSRCDIR%"
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with configuring the build (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

cmake --build . --config %TFBUILD_CONFIG% --target install
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with the build (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

cd %current_dir%

echo "*TF* ***************** Tissue Forge core build: end *****************"

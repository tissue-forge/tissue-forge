@echo off

echo "*TF* **********************************************"
echo "*TF* Launching Tissue Forge conda build for Windows"
echo "*TF* **********************************************"

set TFBUILD_CONFIG=Release

set TFPACKAGELOCALOFF=1
set TFPACKAGECONDA=1

echo "*TF* Executing Tissue Forge local build with the following parameters "
echo "*TF* TFBUILD_CONFIG %TFBUILD_CONFIG%"
echo "*TF* TFPACKAGELOCALOFF %TFPACKAGELOCALOFF%"
echo "*TF* TFPACKAGECONDA %TFPACKAGECONDA%"
echo "*TF* PREFIX %PREFIX%"
echo "*TF* LIBRARY_PREFIX %LIBRARY_PREFIX%"
echo "*TF* SP_DIR %SP_DIR%"
echo "*TF* PYTHON %PYTHON%"
echo "*TF* SRC_DIR %SRC_DIR%"
echo "*TF* ****************************************************************"

mkdir tf_build_conda
cd tf_build_conda

echo "*TF* ****************************************************************"

cmake -DCMAKE_BUILD_TYPE:STRING="%TFBUILD_CONFIG%" ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH="%PREFIX%" ^
      -DCMAKE_FIND_ROOT_PATH="%LIBRARY_PREFIX%" ^
      -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
      -DTF_INSTALL_PREFIX_PYTHON:PATH="%SP_DIR%" ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPThreads_ROOT:PATH="%LIBRARY_PREFIX%" ^
      -DPython_EXECUTABLE=%PYTHON% ^
      -DLIBXML_INCLUDE_DIR:PATH="%LIBRARY_PREFIX%\include\libxml2" ^
      "%SRC_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with configuring the build (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

cmake --build . --config "%TFBUILD_CONFIG%" --target install
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with the build (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

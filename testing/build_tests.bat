@echo off

echo "*TF* **********************************************"
echo "*TF* Launching Tissue Forge tests build for Windows"
echo "*TF* **********************************************"

set current_dir=%cd%

if not exist "%TFENV%" (
    echo "*TF* Environment not found (TFENV=%TFENV%)"
    exit /B 1
)

if not exist "%TFINSTALLDIR%" (
    echo "*TF* Installation not found (TFINSTALLDIR=%TFINSTALLDIR%)"
    exit /B 1
)

set TFTESTS_BUILDDIR=%~dp0build

mkdir %TFTESTS_BUILDDIR%

cd %~dp0

cmake -DCMAKE_BUILD_TYPE:STRING=%TFBUILD_CONFIG% ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH:PATH="%TFENV%;%TFINSTALLDIR%;%TFINSTALLDIR%/lib" ^
      -DCMAKE_FIND_ROOT_PATH:PATH=%TFENV%\Library ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPython_EXECUTABLE:PATH=%TFENV%\python.exe ^
      -S . ^
      -B "%TFTESTS_BUILDDIR%"
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with configuring the build (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

cmake --build "%TFTESTS_BUILDDIR%" --config %TFBUILD_CONFIG%
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with the build (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

cd %current_dir%

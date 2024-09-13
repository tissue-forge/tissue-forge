@echo off

echo "*TF* ****************************************"
echo "*TF* Launching Tissue Forge tests for Windows"
echo "*TF* ****************************************"

if not exist "%TFINSTALLDIR%" (
    echo "*TF* Installation not found (TFINSTALLDIR=%TFINSTALLDIR%)"
    exit /B 1
)

set TFTESTS_TESTSDIR=%~dp0build
set PATH=%TFINSTALLDIR%/bin;%PATH%
cd %TFTESTS_TESTSDIR%
ctest --output-on-failure

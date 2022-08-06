@echo off

if not exist "%TFINSTALLDIR%" exit 1

set TFTESTS_TESTSDIR=%~dp0build
set PATH=%TFINSTALLDIR%/bin;%PATH%
cd %TFTESTS_TESTSDIR%
ctest --output-on-failure

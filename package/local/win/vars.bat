@echo off

call %~dp0site_vars

if not exist "%TFPYSITEDIR%" (
    echo "*TF* site-packages not found (TFPYSITEDIR=%TFPYSITEDIR%)"
    exit /B 1
)
if not exist "%TFENV%" (
    echo "*TF* Environment not found (TFENV=%TFENV%)"
    exit /B 1
)

set PYTHONPATH=%TFPYSITEDIR%;%PYTHONPATH%

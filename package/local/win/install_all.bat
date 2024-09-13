@echo off

set current_dir=%cd%

if not exist "%TFSRCDIR%" (
    echo "*TF* Source directory not found (TFSRCDIR="%TFSRCDIR%")"
    exit /B 1
)

call %TFSRCDIR%\package\local\win\install_core
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with installing core (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

cd %current_dir%

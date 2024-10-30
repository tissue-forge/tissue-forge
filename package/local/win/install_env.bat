@echo off

if not exist "%TFSRCDIR%" (
    echo "*TF* Source not found (TFSRCDIR=%TFSRCDIR%)"
    exit /B 1
)

call conda create --yes --prefix %TFENV%
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong when creating the environment (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

call conda env update --prefix %TFENV% --file %TFSRCDIR%\package\local\win\env.yml
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong when populating the environment (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

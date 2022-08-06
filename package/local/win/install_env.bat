@echo off

if not exist "%TFSRCDIR%" exit 1

call conda create --yes --prefix %TFENV%
if errorlevel 1 exit 2

call conda env update --prefix %TFENV% --file %TFSRCDIR%\package\local\win\env.yml
if errorlevel 1 exit 3

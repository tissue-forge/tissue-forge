@echo off

set TFENV=%~dp0env

call conda create --yes --prefix %TFENV%
if errorlevel 1 exit 1

call conda env update --prefix %TFENV% --file %~dp0rtenv.yml
if errorlevel 1 exit 1

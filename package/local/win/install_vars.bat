@echo off

rem build configuration
set TFBUILD_CONFIG=Release

rem path to source root
set TFSRCDIR=%~dp0..\..\..

rem path to build root
set TFBUILDDIR=%TFSRCDIR%\..\tissue-forge_build

rem path to install root
set TFINSTALLDIR=%TFSRCDIR%\..\tissue-forge_install

rem path to environment root
set TFENV=%TFINSTALLDIR%\env

rem local build qualifier
set TFBUILDQUAL=local

rem path to cuda root directory
set TFCUDAENV=""

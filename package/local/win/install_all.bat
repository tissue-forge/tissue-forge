@echo off

set current_dir=%cd%

if not exist "%TFSRCDIR%" exit 1

call %TFSRCDIR%\package\local\win\install_core
if errorlevel 1 exit 2

cd %current_dir%

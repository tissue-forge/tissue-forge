@echo off

call %~dp0site_vars

if not exist "%TFPYSITEDIR%" exit 1

set PYTHONPATH=%TFPYSITEDIR%;%PYTHONPATH%

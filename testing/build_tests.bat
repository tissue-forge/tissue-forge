@echo off

set current_dir=%cd%

if not exist "%TFENV%" exit 1

if not exist "%TFINSTALLDIR%" exit 1

set TFTESTS_BUILDDIR=%~dp0build

mkdir %TFTESTS_BUILDDIR%

cd %~dp0

cmake -DCMAKE_BUILD_TYPE:STRING=%TFBUILD_CONFIG% ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH:PATH="%TFENV%;%TFINSTALLDIR%;%TFINSTALLDIR%/lib" ^
      -DCMAKE_FIND_ROOT_PATH:PATH=%TFENV%\Library ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPython_EXECUTABLE:PATH=%TFENV%\python.exe ^
      -DPThreads_ROOT:PATH=%TFENV%\Library ^
      -S . ^
      -B "%TFTESTS_BUILDDIR%"
if errorlevel 1 exit 2

cmake --build "%TFTESTS_BUILDDIR%" --config %TFBUILD_CONFIG%
if errorlevel 1 exit 3

cd %current_dir%

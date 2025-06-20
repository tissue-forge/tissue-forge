@echo off

if not exist "%TFSRCDIR%" exit 1
if not exist "%TFENV%" exit 2

set current_dir=%cd%

mkdir %TFBUILDDIR%
mkdir %TFINSTALLDIR%

cd %TFBUILDDIR%

set CC=clang-cl
set CXX=clang-cl

cmake -DCMAKE_BUILD_TYPE:STRING=%TFBUILD_CONFIG% ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH:PATH=%TFENV% ^
      -DCMAKE_FIND_ROOT_PATH:PATH=%TFENV%\Library ^
      -DCMAKE_INSTALL_PREFIX:PATH=%TFINSTALLDIR% ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPython_EXECUTABLE:PATH=%TFENV%\python.exe ^
      -DLIBXML_INCLUDE_DIR:PATH=%TFENV%\Library\include\libxml2 ^
      -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=%TFCUDAENV% ^
      "%TFSRCDIR%"
if errorlevel 1 exit 3

cmake --build . --config %TFBUILD_CONFIG% --target install
if errorlevel 1 exit 4

cd %current_dir%

set TFBUILD_CONFIG=Release

set TFPACKAGELOCALOFF=1
set TFPACKAGECONDA=1

mkdir tf_build_conda
cd tf_build_conda

cmake -DCMAKE_BUILD_TYPE:STRING="%TFBUILD_CONFIG%" ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH="%PREFIX%" ^
      -DCMAKE_FIND_ROOT_PATH="%LIBRARY_PREFIX%" ^
      -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
      -DTF_INSTALL_PREFIX_PYTHON:PATH="%SP_DIR%" ^
      -DCMAKE_C_COMPILER:PATH="%LIBRARY_PREFIX%\bin\clang-cl.exe" ^
      -DCMAKE_CXX_COMPILER:PATH="%LIBRARY_PREFIX%\bin\clang-cl.exe" ^
      -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld ^
      -DPThreads_ROOT:PATH="%LIBRARY_PREFIX%" ^
      -DPython_EXECUTABLE=%PYTHON% ^
      -DLIBXML_INCLUDE_DIR:PATH="%LIBRARY_PREFIX%\include\libxml2" ^
      "%SRC_DIR%"
if errorlevel 1 exit 1

cmake --build . --config "%TFBUILD_CONFIG%" --target install
if errorlevel 1 exit 1

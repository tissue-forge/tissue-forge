@echo off

set current_dir=%cd%

call %~dp0win\install_vars

if not exist "%TFSRCDIR%" exit 1

call %TFSRCDIR%\package\local\win\install_env

if not defined TF_WITHCUDA goto DoInstall

rem Install CUDA support if requested
if %TF_WITHCUDA% == 1 (
    rem Validate specified compute capability
    if not defined CUDAARCHS (
        echo No compute capability specified
        exit 1
    ) 
    echo Detected CUDA support request
    echo Installing additional dependencies...

    goto SetupCUDA
)

goto DoInstall

:SetupCUDA

    set TFCUDAENV=%TFENV%
    call conda install -y -c nvidia -p %TFENV% cuda>NUL
    goto DoInstall

:DoInstall
    call conda activate %TFENV%>NUL

    call %TFSRCDIR%\package\local\win\install_all
    if errorlevel 1 exit 2

    cd %current_dir%

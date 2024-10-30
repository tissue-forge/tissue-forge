@echo off

echo "*TF* **********************************************"
echo "*TF* Launching Tissue Forge local build for Windows"
echo "*TF* **********************************************"

set current_dir=%cd%

call %~dp0win\install_vars

if not exist "%TFSRCDIR%" (
    echo "*TF* Source directory not found (TFSRCDIR="%TFSRCDIR%")"
    exit /B 1
)

call %TFSRCDIR%\package\local\win\install_env
if %ERRORLEVEL% NEQ 0 (
    echo "*TF* Something went wrong with installing the environment (%ERRORLEVEL%)."
    exit /B %ERRORLEVEL%
)

if not defined TF_WITHCUDA goto DoInstall

rem Install CUDA support if requested
if %TF_WITHCUDA% == 1 (
    rem Validate specified compute capability
    if not defined CUDAARCHS (
        echo "*TF* No compute capability specified"
        exit /B 1
    ) 
    echo "*TF* Detected CUDA support request"
    echo "*TF* Installing additional dependencies..."

    goto SetupCUDA
)

goto DoInstall

:SetupCUDA

    set TFCUDAENV=%TFENV%
    call conda install -y -c nvidia -p %TFENV% cuda>NUL
    if %ERRORLEVEL% NEQ 0 (
        echo "*TF* Something went wrong with installing CUDA (%ERRORLEVEL%)."
        exit /B %ERRORLEVEL%
    )
    goto DoInstall

:DoInstall
    call conda activate %TFENV%>NUL

    call %TFSRCDIR%\package\local\win\install_all
    if %ERRORLEVEL% NEQ 0 (
        echo "*TF* Something went wrong with installing Tissue Forge (%ERRORLEVEL%)."
        exit /B %ERRORLEVEL%
    )

    cd %current_dir%

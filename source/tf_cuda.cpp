/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

#include "tf_cuda.h"
#include <tf_config.h>

#include <tfLogger.h>

#include <cuda_gl_interop.h>

#include <cstdarg>
#include <filesystem>
#include <iostream>
#include <fstream>


#define TF_CUDA_TMP_CURRENTCTX(instr)   \
    bool ic = this->attached;           \
    if(!ic) this->pushCurrent();        \
    instr                               \
    if(!ic) this->popCurrent();


static std::string _tfIncludePath = "";
static std::string _CUDAIncludePath = "";
static std::string _tfResourcePath = "";


using namespace TissueForge;


// CUDARTSource


cuda::CUDARTSource::CUDARTSource(const char *filePath, const char *_name) :
    name{_name}
{
    std::ifstream tf_cuda_ifs(filePath);
    if(!tf_cuda_ifs || !tf_cuda_ifs.good() || tf_cuda_ifs.fail()) tf_exp(std::runtime_error(std::string("Error loading CUDART: ") + _name));

    std::string tf_cuda_s((std::istreambuf_iterator<char>(tf_cuda_ifs)), (std::istreambuf_iterator<char>()));
    tf_cuda_ifs.close();

    TF_Log(LOG_INFORMATION) << "Loaded source: " << filePath;

    this->source = tf_cuda_s;
}

const char *cuda::CUDARTSource::c_str() const {
    return this->source.c_str();
}


// CUDARTProgram


cuda::CUDARTProgram::CUDARTProgram() :
    prog{NULL}, 
    ptx{NULL}, 
    is_compute{true}
{
    cuInit(0);
}

cuda::CUDARTProgram::~CUDARTProgram() {
    if(this->prog) {
        TF_NVRTC_CALL(nvrtcDestroyProgram(this->prog));
        delete this->prog;
        this->prog = NULL;
    }
    if(this->ptx) {
        delete this->ptx;
        this->ptx = NULL;
    }
}

void cuda::CUDARTProgram::addOpt(const std::string &opt) {
    if(this->prog) tf_exp(std::logic_error("Program already compiled."));

    this->opts.push_back(opt);
}

void cuda::CUDARTProgram::addIncludePath(const std::string &ipath) {
    if(this->prog) tf_exp(std::logic_error("Program already compiled."));

    this->includePaths.push_back(ipath);
}

void cuda::CUDARTProgram::addNamedExpr(const std::string &namedExpr) {
    if(this->prog) tf_exp(std::logic_error("Program already compiled."));

    this->namedExprs.push_back(namedExpr);
}

void cuda::CUDARTProgram::compile(const char *src, const char *name, int numHeaders, const char *const *headers, const char *const *includeNames) {
    if(this->prog) tf_exp(std::logic_error("Program already compiled."));

    std::vector<std::string> _opts(this->opts);
    if(this->is_compute) _opts.push_back("--gpu-architecture=compute_" + std::to_string(this->arch));
    else _opts.push_back("--gpu-architecture=sm_" + std::to_string(this->arch));

    std::string _includeOpt = "--include-path=";
    std::vector<std::string> includePaths = {
        cuda::CUDAIncludePath(),
        tfIncludePath(),
        tfPrivateIncludePath()
    };
    for(auto &s : includePaths) 
        if(s.length() > 0) 
            _opts.push_back(_includeOpt + s);
    for(auto &s : this->includePaths) _opts.push_back(_includeOpt + s);
    
    #ifdef TF_CUDA_DEBUG

    _opts.push_back("--device-debug");
    _opts.push_back("--generate-line-info");
    _opts.push_back("--display-error-number");

    #endif

    char **charOpts = new char*[_opts.size()];
    for(int i = 0; i < _opts.size(); i++) {
        charOpts[i] = const_cast<char*>(_opts[i].c_str());

        TF_Log(LOG_INFORMATION) << "Got compile option: " << std::string(charOpts[i]);
    }

    this->prog = new nvrtcProgram();
    TF_NVRTC_CALL(nvrtcCreateProgram(this->prog, src, name, numHeaders, headers, includeNames));

    for(auto ne : this->namedExprs) TF_NVRTC_CALL(nvrtcAddNameExpression(*this->prog, ne.c_str()));

    auto compileResult = nvrtcCompileProgram(*this->prog, _opts.size(), charOpts);
    // Dump log on compile failure
    if(compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(*this->prog, &logSize);
        char *log = new char[logSize];
        nvrtcGetProgramLog(*this->prog, log);
        std::cout << log << std::endl;
        delete[] log;
    }
    TF_NVRTC_CALL(compileResult);

    size_t ptxSize;
    TF_NVRTC_CALL(nvrtcGetPTXSize(*this->prog, &ptxSize));
    this->ptx = new char[ptxSize];
    TF_NVRTC_CALL(nvrtcGetPTX(*this->prog, this->ptx));
}

std::string cuda::CUDARTProgram::loweredName(const std::string namedExpr) {
    if(!this->prog) tf_exp(std::logic_error("Program not compiled."));

    const char *name;
    TF_NVRTC_CALL(nvrtcGetLoweredName(*this->prog, namedExpr.c_str(), &name));

    if(!name) return "";

    TF_Log(LOG_DEBUG) << namedExpr << " -> " << name;

    return std::string(name);
}


// CUDAFunction


cuda::CUDAFunction::CUDAFunction(const std::string &name, cuda::CUDAContext *context) : 
    name{name}, 
    gridDimX{1}, gridDimY{1}, gridDimZ{1}, 
    blockDimX{1}, blockDimY{1}, blockDimZ{1}, 
    sharedMemBytes{0}, hStream{NULL}, extra{NULL}, 
    context{context}
{
    if(!context || !context->module) tf_exp(std::logic_error("No loaded programs."));

    this->function = new CUfunction();
    TF_CUDA_CALL(cuModuleGetFunction(this->function, *context->module, name.c_str()));
}

cuda::CUDAFunction::~CUDAFunction() {
    if(this->function) {
        delete this->function;
        this->function = 0;
    }
}

HRESULT cuda::CUDAFunction::autoConfig(const unsigned int &_nr_arrayElems, 
                                   size_t dynamicSMemSize, 
                                   size_t (*blockSizeToDynamicSMemSize)(int), 
                                   int blockSizeLimit)
{
    int minGridSize;
    int blockSize;
    TF_CUDA_CALL(cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, *this->function, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit));

    if(blockSize == 0) {
        TF_Log(LOG_ERROR) << "Auto-config failed!";

        return E_FAIL;
    }
    this->blockDimX = blockSize;
    this->gridDimX = (_nr_arrayElems + blockSize - 1) / blockSize;
    this->sharedMemBytes = dynamicSMemSize;
    
    TF_Log(LOG_INFORMATION) << "CUDA function " << this->name << " configured...";
    TF_Log(LOG_INFORMATION) << "... array elements : " << _nr_arrayElems;
    if(blockSizeLimit > 0) { TF_Log(LOG_INFORMATION) << "... maximum threads: " << blockSizeLimit; }
    TF_Log(LOG_INFORMATION) << "... block size     : " << this->blockDimX << " threads";
    TF_Log(LOG_INFORMATION) << "... grid size      : " << this->gridDimX << " blocks";
    TF_Log(LOG_INFORMATION) << "... shared memory  : " << this->sharedMemBytes << " B/block";

    return S_OK;
}

void cuda::CUDAFunction::operator()(void **args) {
    TF_CUDA_CALL(cuLaunchKernel(
        *this->function, 
        this->gridDimX, this->gridDimY, this->gridDimZ, 
        this->blockDimX, this->blockDimY, this->blockDimZ, 
        this->sharedMemBytes, this->hStream, args, this->extra
    ));
}

void cuda::CUDAFunction::operator()(int nargs, ...) 
{
    void **args;
    args = (void**)malloc(nargs * sizeof(void*));
    std::va_list argsList;
    va_start(argsList, nargs);
    for(int i = 0; i < nargs; i++) args[i] = va_arg(argsList, void*);
    va_end(argsList);

    (*this)(args);
}


// CUDAContext


cuda::CUDAContext::CUDAContext(CUdevice device) : 
    context{NULL}, 
    device{device}, 
    module{NULL}
{
    this->context = new CUcontext();
    TF_CUDA_CALL(cuCtxCreate(this->context, 0, this->device));
    this->attached = true;
}

cuda::CUDAContext::~CUDAContext() {
    if(this->context) this->destroy();
}

void cuda::CUDAContext::addOpt(CUjit_option opt, void *val) {
    this->compileOpts.push_back(opt);
    this->compileOptVals.push_back(val);
}

void cuda::CUDAContext::loadProgram(const cuda::CUDARTProgram &prog) {
    if(!prog.ptx) tf_exp(std::logic_error("Program not compiled."));

    this->loadPTX(prog.ptx);
}

void cuda::CUDAContext::loadPTX(const char *ptx) {
    if(!this->module) this->module = new CUmodule();

    #ifdef TF_CUDA_DEBUG
    this->addOpt(CU_JIT_GENERATE_DEBUG_INFO, (void*)(size_t)1);
    #endif

    size_t numJitOpts = this->compileOpts.size();
    CUjit_option *jitopts;
    void **jitoptvals;
    if(numJitOpts == 0) {
        jitopts = 0;
        jitoptvals = 0;
    }
    else {
        jitopts = new CUjit_option[numJitOpts];
        jitoptvals = new void*[numJitOpts];
        for(int i = 0; i < numJitOpts; i++) {
            jitopts[i] = this->compileOpts[i];
            jitoptvals[i] = this->compileOptVals[i];

            TF_Log(LOG_INFORMATION) << "Got JIT compile option: " << jitopts[i] << ", " << jitoptvals[i];
        }
    }

    TF_CUDA_TMP_CURRENTCTX(
        TF_CUDA_CALL(cuModuleLoadDataEx(this->module, ptx, numJitOpts, jitopts, jitoptvals));
    )

    if(numJitOpts > 0) {
        delete[] jitopts;
        delete[] jitoptvals;
    }
}

cuda::CUDAFunction *cuda::CUDAContext::getFunction(const char *name) {
    if(!this->module) tf_exp(std::logic_error("No loaded programs."));

    return new cuda::CUDAFunction(name, this);
}

CUdeviceptr *cuda::CUDAContext::getGlobal(const char *name) {
    if(!this->module) tf_exp(std::logic_error("No loaded programs."));

    CUdeviceptr *dptr; 
    size_t bytes;
    TF_CUDA_CALL(cuModuleGetGlobal(dptr, &bytes, *this->module, name));
    return dptr;
}

size_t cuda::CUDAContext::getGlobalSize(const char *name) {
    if(!this->module) tf_exp(std::logic_error("No loaded programs."));

    CUdeviceptr *dptr;
    size_t bytes;
    TF_CUDA_CALL(cuModuleGetGlobal(dptr, &bytes, *this->module, name));
    return bytes;
}

void cuda::CUDAContext::pushCurrent() {
    if(this->attached) tf_exp(std::logic_error("Context already attached."));

    TF_CUDA_CALL(cuCtxPushCurrent(*this->context));
    this->attached = true;
}

CUcontext *cuda::CUDAContext::popCurrent() {
    if(!this->attached) tf_exp(std::logic_error("Context not attached."));

    CUcontext *cu;
    TF_CUDA_CALL(cuCtxPopCurrent(cu));
    this->attached = false;
    return cu;
}

void cuda::CUDAContext::destroy() {
    if(!this->context) tf_exp(std::logic_error("No context to destroy."));

    if(this->module) {
        TF_CUDA_TMP_CURRENTCTX(
            TF_CUDA_CALL(cuModuleUnload(*this->module));
        )
        delete this->module;
        this->module = NULL;
    }

    TF_CUDA_CALL(cuCtxDestroy(*this->context));
    delete this->context;
    this->context = NULL;
    this->attached = false;
}

int cuda::CUDAContext::getAPIVersion() {
    if(!this->context) tf_exp(std::logic_error("No context."));

    unsigned int v;
    TF_CUDA_CALL(cuCtxGetApiVersion(*this->context, &v));
    return v;
}

void cuda::CUDAContext::sync() {
    TF_CUDA_CALL(cuCtxSynchronize());
}


// CUDADevice


cuda::CUDADevice::CUDADevice() :
    device{NULL}
{
    cuInit(0);
}

cuda::CUDADevice::~CUDADevice() {
    if(this->device != NULL) this->detachDevice();
}

void cuda::CUDADevice::attachDevice(const int &deviceId) {
    cuda::CUDADevice::validateDeviceId(deviceId);

    if(this->device != NULL) tf_exp(std::logic_error("Device already attached."));

    this->device = new int();
    TF_CUDA_CALL(cuDeviceGet(this->device, deviceId));
}

void cuda::CUDADevice::detachDevice() {
    this->validateAttached();
    this->device = NULL;
}

std::string cuda::CUDADevice::name() {
    this->validateAttached();
    return cuda::CUDADevice::getDeviceName(*this->device);
}

int cuda::CUDADevice::arch() {
    this->validateAttached();
    int vmaj = this->computeCapabilityMajor();
    int vmin = this->computeCapabilityMinor();
    return vmaj * 10 + vmin;
}

size_t cuda::CUDADevice::totalMem() {
    this->validateAttached();
    return cuda::CUDADevice::getDeviceTotalMem(*this->device);
}

int cuda::CUDADevice::getAttribute(const int &attrib) {
    this->validateAttached();
    return cuda::CUDADevice::getDeviceAttribute(*this->device, attrib);
}

std::string cuda::CUDADevice::PCIBusId() {
    this->validateAttached();
    return cuda::CUDADevice::getDevicePCIBusId(*this->device);
}

cuda::CUDAContext *cuda::CUDADevice::createContext() {
    this->validateAttached();
    return new cuda::CUDAContext(*this->device);
}

cuda::CUDAContext *cuda::CUDADevice::currentContext() {
    this->validateAttached();
    
    cuda::CUDAContext *tfContext;
    
    CUcontext *context = new CUcontext();
    if(TF_CUDA_CALL(cuCtxGetCurrent(context)) != CUDA_SUCCESS) {
        return NULL;
    }

    if(context == NULL) { 
        TF_Log(LOG_TRACE);

        delete context;

        tfContext = new cuda::CUDAContext(*this->device); 
    }
    else {
        TF_Log(LOG_TRACE);

        tfContext = new cuda::CUDAContext();
        tfContext->context = context;
        tfContext->device = *this->device;
        tfContext->attached = true;
    }
    return tfContext;
}

std::string cuda::CUDADevice::getDeviceName(const int &deviceId) {
    cuda::CUDADevice::validateDeviceId(deviceId);

    size_t nameLen = 256;
    char name[nameLen];
    TF_CUDA_CALL(cuDeviceGetName(name, nameLen, deviceId));
    return std::string(name);
}

size_t cuda::CUDADevice::getDeviceTotalMem(const int &deviceId) {
    cuda::CUDADevice::validateDeviceId(deviceId);

    size_t size;
    TF_CUDA_CALL(cuDeviceTotalMem(&size, deviceId));
    return size;
}

int cuda::CUDADevice::getDeviceAttribute(const int &deviceId, const int &attrib) {
    cuda::CUDADevice::validateDeviceId(deviceId);

    int pi;
    CUdevice_attribute attr = (CUdevice_attribute)attrib;
    TF_CUDA_CALL(cuDeviceGetAttribute(&pi, attr, deviceId));
    return pi;
}

int cuda::CUDADevice::getNumDevices() {
    int count;
    TF_CUDA_CALL(cuDeviceGetCount(&count));
    return count;
}

std::string cuda::CUDADevice::getDevicePCIBusId(const int &deviceId) {
    cuda::CUDADevice::validateDeviceId(deviceId);

    char *pciBusId;
    TF_CUDA_CALL(cuDeviceGetPCIBusId(pciBusId, 256, deviceId));
    return pciBusId;
}

int cuda::CUDADevice::getCurrentDevice() {
    int deviceId;
    TF_CUDA_CALL(cuCtxGetDevice(&deviceId));
    return deviceId;
}

void cuda::CUDADevice::validateAttached() {
    if(!this->device) tf_exp(std::logic_error("No device attached."));
}

void cuda::CUDADevice::validateDeviceId(const int &deviceId) {
    if(deviceId < 0 || deviceId > cuda::CUDADevice::getNumDevices())
        tf_exp(std::range_error("Invalid ID selection."));

    return;
}

int cuda::CUDADevice::maxThreadsPerBlock() { return cuda::maxThreadsPerBlock(*this->device); }
int cuda::CUDADevice::maxBlockDimX() { return cuda::maxBlockDimX(*this->device); }
int cuda::CUDADevice::maxBlockDimY() { return cuda::maxBlockDimY(*this->device); }
int cuda::CUDADevice::maxBlockDimZ() { return cuda::maxBlockDimZ(*this->device); }
int cuda::CUDADevice::maxGridDimX() { return cuda::maxGridDimX(*this->device); }
int cuda::CUDADevice::maxGridDimY() { return cuda::maxGridDimY(*this->device); }
int cuda::CUDADevice::maxGridDimZ() { return cuda::maxGridDimZ(*this->device); }
int cuda::CUDADevice::maxSharedMemPerBlock() { return cuda::maxSharedMemPerBlock(*this->device); }
int cuda::CUDADevice::maxTotalMemConst() { return cuda::maxTotalMemConst(*this->device); }
int cuda::CUDADevice::warpSize() { return cuda::warpSize(*this->device); }
int cuda::CUDADevice::maxRegsPerBlock() { return cuda::maxRegsPerBlock(*this->device); }
int cuda::CUDADevice::clockRate() { return cuda::clockRate(*this->device); }
bool cuda::CUDADevice::gpuOverlap() { return cuda::gpuOverlap(*this->device); }
int cuda::CUDADevice::numMultiprocessors() { return cuda::numMultiprocessors(*this->device); }
bool cuda::CUDADevice::kernelExecTimeout() { return cuda::kernelExecTimeout(*this->device); }
bool cuda::CUDADevice::computeModeDefault() { return cuda::computeModeDefault(*this->device); }
bool cuda::CUDADevice::computeModeProhibited() { return cuda::computeModeProhibited(*this->device); }
bool cuda::CUDADevice::computeModeExclusive() { return cuda::computeModeExclusive(*this->device); }
int cuda::CUDADevice::PCIDeviceId() { return cuda::PCIDeviceId(*this->device); }
int cuda::CUDADevice::PCIDomainId() { return cuda::PCIDomainId(*this->device); }
int cuda::CUDADevice::clockRateMem() { return cuda::clockRateMem(*this->device); }
int cuda::CUDADevice::globalMemBusWidth() { return cuda::globalMemBusWidth(*this->device); }
int cuda::CUDADevice::L2CacheSize() { return cuda::L2CacheSize(*this->device); }
int cuda::CUDADevice::maxThreadsPerMultiprocessor() { return cuda::maxThreadsPerMultiprocessor(*this->device); }
int cuda::CUDADevice::computeCapabilityMajor() { return cuda::computeCapabilityMajor(*this->device); }
int cuda::CUDADevice::computeCapabilityMinor() { return cuda::computeCapabilityMinor(*this->device); }
bool cuda::CUDADevice::L1CacheSupportGlobal() { return cuda::L1CacheSupportGlobal(*this->device); }
bool cuda::CUDADevice::L1CacheSupportLocal() { return cuda::L1CacheSupportLocal(*this->device); }
int cuda::CUDADevice::maxSharedMemPerMultiprocessor() { return cuda::maxSharedMemPerMultiprocessor(*this->device); }
int cuda::CUDADevice::maxRegsPerMultiprocessor() { return cuda::maxRegsPerMultiprocessor(*this->device); }
bool cuda::CUDADevice::managedMem() { return cuda::managedMem(*this->device); }
bool cuda::CUDADevice::multiGPUBoard() { return cuda::multiGPUBoard(*this->device); }
int cuda::CUDADevice::multiGPUBoardGroupId() { return cuda::multiGPUBoardGroupId(*this->device); }


// CUDA interface


void cuda::init() {
    cuInit(0);
}

void cuda::setGLDevice(const int &deviceId) {
    cudaGLSetGLDevice(deviceId);
}

std::string cuda::getDeviceName(const int &deviceId) {
    return cuda::CUDADevice::getDeviceName(deviceId);
}

size_t cuda::getDeviceTotalMem(const int &deviceId) {
    return cuda::CUDADevice::getDeviceTotalMem(deviceId);
}

int cuda::getDeviceAttribute(const int &deviceId, const int &attrib) {
    return cuda::CUDADevice::getDeviceAttribute(deviceId, attrib);
}

int cuda::getNumDevices() {
    return cuda::CUDADevice::getNumDevices();
}

std::string cuda::getDevicePCIBusId(const int &deviceId) {
    return cuda::CUDADevice::getDevicePCIBusId(deviceId);
}

int cuda::getCurrentDevice() {
    return cuda::CUDADevice::getCurrentDevice();
}

int cuda::maxThreadsPerBlock(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK); }
int cuda::maxBlockDimX(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X); }
int cuda::maxBlockDimY(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y); }
int cuda::maxBlockDimZ(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z); }
int cuda::maxGridDimX(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X); }
int cuda::maxGridDimY(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y); }
int cuda::maxGridDimZ(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z); }
int cuda::maxSharedMemPerBlock(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK); }
int cuda::maxTotalMemConst(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY); }
int cuda::warpSize(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_WARP_SIZE); }
int cuda::maxRegsPerBlock(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK); }
int cuda::clockRate(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_CLOCK_RATE); }
bool cuda::gpuOverlap(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP); }
int cuda::numMultiprocessors(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT); }
bool cuda::kernelExecTimeout(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT); }
bool cuda::computeModeDefault(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) == CU_COMPUTEMODE_DEFAULT; }
bool cuda::computeModeProhibited(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) == CU_COMPUTEMODE_PROHIBITED; }
bool cuda::computeModeExclusive(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) == CU_COMPUTEMODE_EXCLUSIVE_PROCESS; }
int cuda::PCIDeviceId(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID); }
int cuda::PCIDomainId(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID); }
int cuda::clockRateMem(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE); }
int cuda::globalMemBusWidth(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH); }
int cuda::L2CacheSize(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE); }
int cuda::maxThreadsPerMultiprocessor(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR); }
int cuda::computeCapabilityMajor(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR); }
int cuda::computeCapabilityMinor(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR); }
bool cuda::L1CacheSupportGlobal(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED); }
bool cuda::L1CacheSupportLocal(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED); }
int cuda::maxSharedMemPerMultiprocessor(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR); }
int cuda::maxRegsPerMultiprocessor(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR); }
bool cuda::managedMem(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY); }
bool cuda::multiGPUBoard(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD); }
int cuda::multiGPUBoardGroupId(const int &deviceId) { return cuda::getDeviceAttribute(deviceId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID); }

const char *test_program = "                                                \n\
extern \"C\" __global__ void gpu_test(int len, int *vals, int *result) {    \n\
    int i = blockIdx.x * blockDim.x + threadIdx.x;                          \n\
    if (i >= len) return;                                                   \n\
                                                                            \n\
    int n = 0;                                                              \n\
    int ri = vals[i];                                                       \n\
    for (int j = 0; j < len; ++j)                                           \n\
        if (vals[j] == ri) n++;                                             \n\
    result[i] = n;                                                          \n\
}                                                                           \n";

void cuda::test(const int &numBlocks, const int &numThreads, const int &numEls, const int &deviceId) {
    cuda::init();

    TF_Log(LOG_DEBUG) << "*******************************";
    TF_Log(LOG_DEBUG) << "Starting Tissue Forge CUDA test";
    TF_Log(LOG_DEBUG) << "*******************************";

    TF_Log(LOG_DEBUG) << "Initializing device...";
    
    cuda::CUDADevice device;
    device.attachDevice(deviceId);
    auto ctx = device.createContext();

    int cc_major = device.computeCapabilityMajor();
    int cc_minor = device.computeCapabilityMinor();
    int arch = cc_major * 10 + cc_minor;

    TF_Log(LOG_DEBUG) << "     Number of devices               : " << cuda::getNumDevices();
    TF_Log(LOG_DEBUG) << "     Name of device                  : " << device.name();
    TF_Log(LOG_DEBUG) << "     Compute capability of device    : " << cc_major << "." << cc_minor;
    TF_Log(LOG_DEBUG) << "     Number of threads per block     : " << device.maxThreadsPerBlock();

    TF_Log(LOG_DEBUG) << "JIT compiling program...";

    cuda::CUDARTProgram prog;
    prog.arch = device.arch();
    prog.compile(test_program, "tf_test_program.cu");

    TF_Log(LOG_DEBUG) << "Loading program...";

    ctx->loadProgram(prog);

    TF_Log(LOG_DEBUG) << "Preparing work...";

    size_t bufferSize = numEls * sizeof(float);
    int *vals = (int*)malloc(bufferSize);
    for (int i = 0; i < numEls; ++i) vals[i] = numEls % (i + 1);

    CUstream stream;
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUdeviceptr vals_d;
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuMemAlloc(&vals_d, bufferSize));
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuMemcpyHtoDAsync(vals_d, vals, bufferSize, stream));
    
    int *results = (int*)malloc(bufferSize);
    CUdeviceptr results_d;
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuMemAlloc(&results_d, bufferSize));
    int n = numEls;

    TF_Log(LOG_DEBUG) << "Optimizing kernel...";

    auto f = ctx->getFunction("gpu_test");
    if(f->autoConfig(numEls) != S_OK) {
        TF_Log(LOG_DEBUG) << "    ... failed!" << std::endl;
        f->gridDimX = numBlocks;
        f->blockDimX = numThreads;
    }
    else { TF_Log(LOG_DEBUG) << "    ... done!" << std::endl; }

    TF_Log(LOG_DEBUG) << "Doing work...";

    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuStreamSynchronize(stream));
    void *args[] = {&n, &vals_d, &results_d};
    (*f)(args);
    ctx->sync();

    TF_Log(LOG_DEBUG) << "Retrieving results... " << TF_CUDA_CALL(cuMemcpyDtoHAsync(results, results_d, bufferSize, stream));

    TF_Log(LOG_DEBUG) << "Cleaning up...";
    
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuMemFree(vals_d));
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuMemFree(results_d));
    free(vals);
    free(results);
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuStreamSynchronize(stream));
    TF_Log(LOG_DEBUG) << "    ... " << TF_CUDA_CALL(cuStreamDestroy(stream));
    delete f;
    ctx->destroy();
    device.detachDevice();

    TF_Log(LOG_DEBUG) << "********************************";
    TF_Log(LOG_DEBUG) << "Completed Tissue Forge CUDA test";
    TF_Log(LOG_DEBUG) << "********************************";
}


// Misc.


std::string cuda::tfIncludePath() {
    if(_tfIncludePath.length() == 0) 
        return "";
    auto p = std::filesystem::absolute(_tfIncludePath);
    return p.string();
}

HRESULT cuda::setTfIncludePath(const std::string &_path) {
    _tfIncludePath = _path;
    return S_OK;
}

std::string cuda::tfPrivateIncludePath() {
    std::string ip = tfIncludePath();
    if(ip.length() == 0) 
        return "";
    auto p = std::filesystem::absolute(ip);
    p.append("private");
    return p.string();
}

std::string cuda::tfResourcePath() {
    return _tfResourcePath;
}

HRESULT cuda::setTfResourcePath(const std::string &_path) {
    _tfResourcePath = _path;
    return S_OK;
}

std::string cuda::CUDAPath() {
    if(_tfResourcePath.length() == 0) 
        return "";
    auto p = std::filesystem::absolute(_tfResourcePath);
    p.append("cuda");
    return p.string();
}

std::string cuda::CUDAIncludePath() {
    return _CUDAIncludePath;
}

HRESULT cuda::setCUDAIncludePath(const std::string &_path) {
    _CUDAIncludePath = _path;
    return S_OK;
}

std::string cuda::CUDAResourcePath(const std::string &relativePath) {
    std::string cp = cuda::CUDAPath();
    if(cp.length() == 0) 
        return "";
    auto p = std::filesystem::absolute(cp);
    p.append(relativePath);
    return p.string();
}

std::string cuda::CUDAPTXObjectRelPath() {
    return std::string("objects-") + std::string(TF_BUILD_TYPE);
}

std::vector<std::string> cuda::CUDAArchs() {
    char s[] = TF_CUDA_ARCHS;
    char *token = strtok(s, ";");
    std::vector<std::string> result;
    while(token != NULL) {
        result.push_back(std::string(token));
        token = strtok(s, ";");
    }
    return result;
}

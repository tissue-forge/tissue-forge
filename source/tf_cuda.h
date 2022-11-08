/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

/**
 * @file tf_cuda.h
 * 
 */

// TODO: implement support for JIT-compiled programs and kernel usage in wrapped languages

#ifndef _SOURCE_TF_CUDA_H_
#define _SOURCE_TF_CUDA_H_

#include "tfError.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <vector>
#include <stdexcept>
#include <string>


namespace TissueForge::cuda {


    enum ErrorCode : int {
        TFCUDAERR_ok = 0,
        TFCUDAERR_setdevice,
        TFCUDAERR_setblocks,
        TFCUDAERR_setthreads,
        TFCUDAERR_ondevice,
        TFCUDAERR_notondevice,
        TFCUDAERR_cleardevices,
        TFCUDAERR_refresh,
        TFCUDAERR_send,
        TFCUDAERR_pull,
        TFCUDAERR_LAST
    };

    /* list of error messages. */
    static const char *tfcuda_err_msg[TFCUDAERR_LAST] = {
        "No CUDA errors.",
        "Failed to set device.",
        "Failed to set blocks.",
        "Failed to set threads.",
        "Already on device.",
        "Not on device.",
        "Failed to clear devices.",
        "Refresh failed.",
        "Attempting send to device failed.",
        "Attempting pull from device when not sent."
    };

    inline CUresult cuda_errorchk(CUresult retCode, const char *file, int line) {
        if(retCode != CUDA_SUCCESS) {
            std::string msg = "CUDA failed with error: ";
            const char *cmsg;
            cuGetErrorName(retCode, &cmsg);
            msg += std::string(cmsg);
            msg += ", " + std::string(file) + ", " + std::to_string(line);
            tf_exp(std::runtime_error(msg.c_str()));
        }
        return retCode;
    }
    #ifndef TF_CUDA_CALL
        #define TF_CUDA_CALL(res) cuda_errorchk(res, __FILE__, __LINE__)
    #endif

    inline nvrtcResult nvrtc_errorchk(nvrtcResult retCode, const char *file, int line) {
        if(retCode != NVRTC_SUCCESS) {
            std::string msg = "NVRTC failed with error: ";
            msg += std::string(nvrtcGetErrorString(retCode));
            msg += ", " + std::string(file) + ", " + std::to_string(line);
            tf_exp(std::runtime_error(msg.c_str()));
        }
        return retCode;
    }
    #ifndef TF_NVRTC_CALL
        #define TF_NVRTC_CALL(res) nvrtc_errorchk(res, __FILE__, __LINE__)
    #endif

    inline cudaError_t cudart_errorchk(cudaError_t retCode, const char *file, int line) {
        if(retCode != cudaSuccess) {
            std::string msg = "NVRTC failed with error: ";
            msg += std::string(cudaGetErrorString(retCode));
            msg += ", " + std::string(file) + ", " + std::to_string(line);
            tf_exp(std::runtime_error(msg.c_str()));
        }
        return retCode;
    }
    #ifndef TF_CUDART_CALL
        #define TF_CUDART_CALL(res) cudart_errorchk(res, __FILE__, __LINE__)
    #endif


    /**
     * @brief Convenience class for loading source from file and storing, here intended for CUDA
     * 
     */
    struct CUDARTSource {
        std::string source;
        const char *name;

        CUDARTSource(const char *filePath, const char *_name);
        const char *c_str() const;
    };


    /**
     * @brief A JIT-compiled CUDA Tissue Forge program. 
     * 
     * This object wraps the procedures for turning CUDA source into executable kernels at runtime 
     * using NVRTC. 
     * 
     */
    struct CUDARTProgram {

        nvrtcProgram *prog;
        char *ptx;
        std::vector<std::string> opts;
        std::vector<std::string> namedExprs;
        std::vector<std::string> includePaths;
        int arch;
        bool is_compute;

        CUDARTProgram();
        ~CUDARTProgram();

        /**
         * @brief Add a compilation option
         * 
         * @param opt 
         */
        void addOpt(const std::string &opt);

        /**
         * @brief Add a directory to include in the search path
         * 
         * @param ipath 
         */
        void addIncludePath(const std::string &ipath);

        /**
         * @brief Add a named expression
         * 
         * @param namedExpr 
         */
        void addNamedExpr(const std::string &namedExpr);

        /**
         * @brief Compile the program
         * 
         * @param src 
         * @param name 
         * @param numHeaders 
         * @param headers 
         * @param includeNames 
         */
        void compile(const char *src, const char *name, int numHeaders=0, const char *const *headers=0, const char *const *includeNames=0);

        /**
         * @brief Get the lowered name of a named expression. Cannot be called until after compilaton. 
         * 
         * @param namedExpr 
         * @return std::string 
         */
        std::string loweredName(const std::string namedExpr);

    };

    struct CUDAContext;


    /**
     * @brief A CUDA kernel from a JIT-compiled Tissue Forge program
     * 
     */
    struct CUDAFunction {
        const std::string name;
        unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
        CUstream hStream;
        void **extra;

        CUDAFunction(const std::string &name, CUDAContext *context);
        ~CUDAFunction();

        HRESULT autoConfig(const unsigned int &_nr_arrayElems, 
                        size_t dynamicSMemSize=0, 
                        size_t (*blockSizeToDynamicSMemSize)(int)=0, 
                        int blockSizeLimit=0);

        void operator()(void **args);
        void operator()(int nargs, ...);

    private:
        CUfunction *function;
        CUDAContext *context;
    };


    /**
     * @brief A convenience wrap of the CUDA context for JIT-compiled Tissue Forge programs. 
     * 
     */
    struct CUDAContext {

        CUcontext *context;
        CUdevice device;
        CUmodule *module;
        std::vector<CUjit_option> compileOpts;
        std::vector<void*> compileOptVals;

        /* Flag signifying whether this context is attached to a CPU thread. */
        bool attached;

        CUDAContext(CUdevice device=0);
        ~CUDAContext();

        void addOpt(CUjit_option opt, void *val);

        /**
         * @brief Load a compiled program. 
         * 
         * @param prog the compiled program
         */
        void loadProgram(const CUDARTProgram &prog);

        /**
         * @brief Load pre-compiled ptx
         * 
         * @param ptx 
         */
        void loadPTX(const char *ptx);

        /**
         * @brief Get a cuda function from a loaded module. 
         * 
         * @param name 
         * @return CUDAFunction* 
         */
        CUDAFunction *getFunction(const char *name);

        /**
         * @brief Get a global pointer from a loaded module. 
         * 
         * @param name 
         * @return CUdeviceptr* 
         */
        CUdeviceptr *getGlobal(const char *name);

        /**
         * @brief Get the size of a global pointer from a loaded module. 
         * 
         * @param name 
         * @return size_t 
         */
        size_t getGlobalSize(const char *name);

        /**
         * @brief Push the context onto the stack of current contexts of the CPU thread. 
         * 
         * The context becomes the current context of the CPU thread. 
         * 
         */
        void pushCurrent();

        /**
         * @brief Pop the context from the stack and returns the new current context of contexts of the CPU thread. 
         * 
         * After being popped, the context can be pushed to a different CPU thread. 
         * 
         * @return CUcontext* 
         */
        CUcontext *popCurrent();

        /**
         * @brief Destroy the context. 
         * 
         */
        void destroy();

        /**
         * @brief Get the API version of this context. 
         * 
         * @return int 
         */
        int getAPIVersion();

        /**
         * @brief Synchronize GPU with calling CPU thread. Blocks until all preceding tasks of the current context are complete. 
         * 
         */
        static void sync();
    };


    /**
     * @brief A simple interface with a CUDA device
     * 
     */
    struct CUDADevice {

        CUdevice *device;

        CUDADevice();
        ~CUDADevice();
        
        /**
         * @brief Attach a CUDA-supporting device by id. 
         * 
         * @param deviceId id of device
         */
        void attachDevice(const int &deviceId=0);

        /**
         * @brief Detach currently attached device. 
         * 
         */
        void detachDevice();

        /**
         * @brief Get the name of attached device
         * 
         * @return std::string 
         */
        std::string name();

        /**
         * @brief Get architecture of attached device
         * 
         * @return int 
         */
        int arch();

        /**
         * @brief Get the total memory of attached device
         * 
         * @return size_t 
         */
        size_t totalMem();

        /**
         * @brief Get the attribute value of attached device
         * 
         * @param attrib 
         * @return int 
         */
        int getAttribute(const int &attrib);

        /**
         * @brief Get the PCI bus id of this device. 
         * 
         * @return std::string 
         */
        std::string PCIBusId();

        /**
         * @brief Create a context on this device. 
         * 
         * Calling thread is responsible for destroying context. 
         * 
         * @return CUDAContext* 
         */
        CUDAContext *createContext();

        /**
         * @brief Get the current context. If none exists, one is created. 
         * 
         * @return CUDAContext* 
         */
        CUDAContext *currentContext();

        /**
         * @brief Get the name of a device
         * 
         * @param deviceId 
         * @return std::string 
         */
        static std::string getDeviceName(const int &deviceId);

        /**
         * @brief Get the total memory of device
         * 
         * @param deviceId 
         * @return size_t 
         */
        static size_t getDeviceTotalMem(const int &deviceId);

        /**
         * @brief Get the attribute value of a device
         * 
         * @param deviceId 
         * @param attrib 
         * @return int 
         */
        static int getDeviceAttribute(const int &deviceId, const int &attrib);

        /**
         * @brief Get number of available compute-capable devices
         * 
         * @return int 
         */
        static int getNumDevices();

        /**
         * @brief Get the PCI bus id of a device
         * 
         * @param deviceId 
         * @return std::string 
         */
        static std::string getDevicePCIBusId(const int &deviceId);

        /**
         * @brief Get the device id of the current context of the calling CPU thread. 
         * 
         * @return int 
         */
        static int getCurrentDevice();

        /**
         * @brief Maximum number of threads per block
         * 
         * @return int 
         */
        int maxThreadsPerBlock();

        /**
         * @brief Maximum x-dimension of a block
         * 
         * @return int 
         */
        int maxBlockDimX();

        /**
         * @brief Maximum y-dimension of a block
         * 
         * @return int 
         */
        int maxBlockDimY();

        /**
         * @brief Maximum z-dimension of a block
         * 
         * @return int 
         */
        int maxBlockDimZ();

        /**
         * @brief Maximum x-dimension of a grid
         * 
         * @return int 
         */
        int maxGridDimX();

        /**
         * @brief Maximum y-dimension of a grid
         * 
         * @return int 
         */
        int maxGridDimY();

        /**
         * @brief Maximum z-dimension of a grid
         * 
         * @return int 
         */
        int maxGridDimZ();

        /**
         * @brief Maximum amount of shared memory available to a thread block in bytes
         * 
         * @return int 
         */
        int maxSharedMemPerBlock();

        /**
         * @brief Memory available on device for __constant__ variables in a CUDA C kernel in bytes
         * 
         * @return int 
         */
        int maxTotalMemConst();

        /**
         * @brief Warp size in threads
         * 
         * @return int 
         */
        int warpSize();

        /**
         * @brief Maximum number of 32-bit registers available to a thread block
         * 
         * @return int 
         */
        int maxRegsPerBlock();

        /**
         * @brief The typical clock frequency in kilohertz
         * 
         * @return int 
         */
        int clockRate();

        /**
         * @brief Test if the device can concurrently copy memory between host and device while executing a kernel
         * 
         * @return true 
         * @return false 
         */
        bool gpuOverlap();

        /**
         * @brief Number of multiprocessors on the device
         * 
         * @return int 
         */
        int numMultiprocessors();

        /**
         * @brief Test if there is a run time limit for kernels executed on the device
         * 
         * @return true 
         * @return false 
         */
        bool kernelExecTimeout();

        /**
         * @brief Test if device is not restricted and can have multiple CUDA contexts present at a single time
         * 
         * @return true 
         * @return false 
         */
        bool computeModeDefault();

        /**
         * @brief Test if device is prohibited from creating new CUDA contexts
         * 
         * @return true 
         * @return false 
         */
        bool computeModeProhibited();

        /**
         * @brief Test if device can have only one context used by a single process at a time
         * 
         * @return true 
         * @return false 
         */
        bool computeModeExclusive();

        /**
         * @brief PCI device (also known as slot) identifier of the device
         * 
         * @return int 
         */
        int PCIDeviceId();

        /**
         * @brief PCI domain identifier of the device
         * 
         * @return int 
         */
        int PCIDomainId();

        /**
         * @brief Peak memory clock frequency in kilohertz
         * 
         * @return int 
         */
        int clockRateMem();

        /**
         * @brief Global memory bus width in bits
         * 
         * @return int 
         */
        int globalMemBusWidth();

        /**
         * @brief Size of L2 cache in bytes. 0 if the device doesn't have L2 cache
         * 
         * @return int 
         */
        int L2CacheSize();

        /**
         * @brief Maximum resident threads per multiprocessor
         * 
         * @return int 
         */
        int maxThreadsPerMultiprocessor();

        /**
         * @brief Major compute capability version number
         * 
         * @return int 
         */
        int computeCapabilityMajor();

        /**
         * @brief Minor compute capability version number
         * 
         * @return int 
         */
        int computeCapabilityMinor();

        /**
         * @brief Test if device supports caching globals in L1 cache
         * 
         * @return true 
         * @return false 
         */
        bool L1CacheSupportGlobal();

        /**
         * @brief Test if device supports caching locals in L1 cache
         * 
         * @return true 
         * @return false 
         */
        bool L1CacheSupportLocal();

        /**
         * @brief Maximum amount of shared memory available to a multiprocessor in bytes
         * 
         * @return int 
         */
        int maxSharedMemPerMultiprocessor();

        /**
         * @brief Maximum number of 32-bit registers available to a multiprocessor
         * 
         * @return int 
         */
        int maxRegsPerMultiprocessor();

        /**
         * @brief Test if device supports allocating managed memory on this system
         * 
         * @return true 
         * @return false 
         */
        bool managedMem();

        /**
         * @brief Test if device is on a multi-GPU board
         * 
         * @return true 
         * @return false 
         */
        bool multiGPUBoard();

        /**
         * @brief Unique identifier for a group of devices associated with the same board
         * 
         * @return int 
         */
        int multiGPUBoardGroupId();

    private:
        
        void validateAttached();
        static void validateDeviceId(const int &deviceId);
    };


    // Tissue Forge CUDA interface


    /**
     * @brief Initialize CUDA
     * 
     */
    CPPAPI_FUNC(void) init();

    CPPAPI_FUNC(void) setGLDevice(const int &deviceId);

    /**
     * @brief Get the name of a device
     * 
     * @param deviceId 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) getDeviceName(const int &deviceId);

    /**
     * @brief Get the total memory of device
     * 
     * @param deviceId 
     * @return size_t 
     */
    CPPAPI_FUNC(size_t) getDeviceTotalMem(const int &deviceId);

    /**
     * @brief Get the attribute value of a device
     * 
     * @param deviceId 
     * @param attrib 
     * @return int 
     */
    CPPAPI_FUNC(int) getDeviceAttribute(const int &deviceId, const int &attrib);

    /**
     * @brief Get number of available compute-capable devices
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) getNumDevices();

    /**
     * @brief Get the PCI bus id of a device
     * 
     * @param deviceId 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) getDevicePCIBusId(const int &deviceId);

    /**
     * @brief Get the device id of the current context of the calling CPU thread. 
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) getCurrentDevice();

    /**
     * @brief Maximum number of threads per block
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxThreadsPerBlock(const int &deviceId);

    /**
     * @brief Maximum x-dimension of a block
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxBlockDimX(const int &deviceId);

    /**
     * @brief Maximum y-dimension of a block
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxBlockDimY(const int &deviceId);

    /**
     * @brief Maximum z-dimension of a block
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxBlockDimZ(const int &deviceId);

    /**
     * @brief Maximum x-dimension of a grid
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxGridDimX(const int &deviceId);

    /**
     * @brief Maximum y-dimension of a grid
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxGridDimY(const int &deviceId);

    /**
     * @brief Maximum z-dimension of a grid
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxGridDimZ(const int &deviceId);

    /**
     * @brief Maximum amount of shared memory available to a thread block in bytes
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxSharedMemPerBlock(const int &deviceId);

    /**
     * @brief Memory available on device for __constant__ variables in a CUDA C kernel in bytes
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxTotalMemConst(const int &deviceId);

    /**
     * @brief Warp size in threads
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) warpSize(const int &deviceId);

    /**
     * @brief Maximum number of 32-bit registers available to a thread block
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxRegsPerBlock(const int &deviceId);

    /**
     * @brief The typical clock frequency in kilohertz
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) clockRate(const int &deviceId);

    /**
     * @brief Test if the device can concurrently copy memory between host and device while executing a kernel
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) gpuOverlap(const int &deviceId);

    /**
     * @brief Number of multiprocessors on the device
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) numMultiprocessors(const int &deviceId);

    /**
     * @brief Test if there is a run time limit for kernels executed on the device
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) kernelExecTimeout(const int &deviceId);

    /**
     * @brief Test if device is not restricted and can have multiple CUDA contexts present at a single time
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) computeModeDefault(const int &deviceId);

    /**
     * @brief Test if device is prohibited from creating new CUDA contexts
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) computeModeProhibited(const int &deviceId);

    /**
     * @brief Test if device can have only one context used by a single process at a time
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) computeModeExclusive(const int &deviceId);

    /**
     * @brief PCI device (also known as slot) identifier of the device
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) PCIDeviceId(const int &deviceId);

    /**
     * @brief PCI domain identifier of the device
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) PCIDomainId(const int &deviceId);

    /**
     * @brief Peak memory clock frequency in kilohertz
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) clockRateMem(const int &deviceId);

    /**
     * @brief Global memory bus width in bits
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) globalMemBusWidth(const int &deviceId);

    /**
     * @brief Size of L2 cache in bytes. 0 if the device doesn't have L2 cache
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) L2CacheSize(const int &deviceId);

    /**
     * @brief Maximum resident threads per multiprocessor
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxThreadsPerMultiprocessor(const int &deviceId);

    /**
     * @brief Major compute capability version number
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) computeCapabilityMajor(const int &deviceId);

    /**
     * @brief Minor compute capability version number
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) computeCapabilityMinor(const int &deviceId);

    /**
     * @brief Test if device supports caching globals in L1 cache
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) L1CacheSupportGlobal(const int &deviceId);

    /**
     * @brief Test if device supports caching locals in L1 cache
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) L1CacheSupportLocal(const int &deviceId);

    /**
     * @brief Maximum amount of shared memory available to a multiprocessor in bytes
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxSharedMemPerMultiprocessor(const int &deviceId);

    /**
     * @brief Maximum number of 32-bit registers available to a multiprocessor
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) maxRegsPerMultiprocessor(const int &deviceId);

    /**
     * @brief Test if device supports allocating managed memory on this system
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) managedMem(const int &deviceId);

    /**
     * @brief Test if device is on a multi-GPU board
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) multiGPUBoard(const int &deviceId);

    /**
     * @brief Unique identifier for a group of devices associated with the same board
     * 
     * @return int 
     */
    CPPAPI_FUNC(int) multiGPUBoardGroupId(const int &deviceId);

    /**
     * @brief Tests JIT-compiled program execution and deployment. 
     * 
     * Enable logger at debug level to see step-by-step report. 
     * 
     * @param numBlocks number of blocks
     * @param numThreads number of threads
     * @param numEls number of elements in calculations
     * @param deviceId ID of CUDA device
     */
    CPPAPI_FUNC(void) test(const int &numBlocks, const int &numThreads, const int &numEls, const int &deviceId=0);


    /**
     * @brief Returns the current path to the installed Tissue Forge include directory
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) tfIncludePath();
    
    /**
     * @brief Set the current path to the installed Tissue Forge include directory
     * 
     * @param _path absolute path
     * @return HRESULT
     */
    CPPAPI_FUNC(HRESULT) setTfIncludePath(const std::string &_path);

    /**
     * @brief Returns the path to the installed Tissue Forge private include directory
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) tfPrivateIncludePath();

    /**
     * @brief Returns the current path to the installed Tissue Forge resource directory
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) tfResourcePath();

    /**
     * @brief Set the current path to the installed Tissue Forge resource directory
     * 
     * @param _path absolute path
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setTfResourcePath(const std::string &_path);

    /**
     * @brief Returns the path to the installed Tissue Forge CUDA resources directory
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) CUDAPath();

    /**
     * @brief Returns the current path to the installed CUDA include directory
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) CUDAIncludePath();

    /**
     * @brief Set the current path to the installed CUDA include directory
     * 
     * @param _path absolute path
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setCUDAIncludePath(const std::string &_path);

    /**
     * @brief Returns an absolute path to a subdirectory of the install Tissue Forge CUDA resources directory
     * 
     * @param relativePath 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) CUDAResourcePath(const std::string &relativePath);

    /**
     * @brief Returns the relative path to the installed Tissue Forge CUDA PTX object directory. 
     * 
     * The path is relative to the Tissue Forge CUDA resources directory, and depends on the build type of the installation. 
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) CUDAPTXObjectRelPath();

    /**
     * @brief Returns the supported CUDA architectures of the installation. 
     * 
     * @return std::vector<std::string> 
     */
    CPPAPI_FUNC(std::vector<std::string>) CUDAArchs();

};

#endif // _SOURCE_TF_CUDA_H_
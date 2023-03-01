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

%{

#include "tf_cuda.h"

#include "cuda/tfSimulatorConfig.h"
#include "cuda/tfAngleConfig.h"
#include "cuda/tfBondConfig.h"
#include "cuda/tfEngineConfig.h"

%}


%ignore cuda_errorchk;
%ignore nvrtc_errorchk;
%ignore cudart_errorchk;

%ignore CUDARTSource;
%ignore CUDARTProgram;
%ignore CUDAFunction;
%ignore CUDAContext;
%ignore CUDADevice;

%rename(on_device) TissueForge::cuda::AngleConfig::onDevice;
%rename(get_device) TissueForge::cuda::AngleConfig::getDevice;
%rename(to_device) TissueForge::cuda::AngleConfig::toDevice;
%rename(from_device) TissueForge::cuda::AngleConfig::fromDevice;
%rename(set_blocks) TissueForge::cuda::AngleConfig::setBlocks;
%rename(set_threads) TissueForge::cuda::AngleConfig::setThreads;
%rename(refresh_angle) TissueForge::cuda::AngleConfig::refreshAngle;
%rename(refresh_angles) TissueForge::cuda::AngleConfig::refreshAngles;
%rename(on_device) TissueForge::cuda::BondConfig::onDevice;
%rename(get_device) TissueForge::cuda::BondConfig::getDevice;
%rename(set_device) TissueForge::cuda::BondConfig::setDevice;
%rename(to_device) TissueForge::cuda::BondConfig::toDevice;
%rename(from_device) TissueForge::cuda::BondConfig::fromDevice;
%rename(set_blocks) TissueForge::cuda::BondConfig::setBlocks;
%rename(set_threads) TissueForge::cuda::BondConfig::setThreads;
%rename(refresh_bond) TissueForge::cuda::BondConfig::refreshBond;
%rename(refresh_bonds) TissueForge::cuda::BondConfig::refreshBonds;
%rename(on_device) TissueForge::cuda::EngineConfig::onDevice;
%rename(get_device) TissueForge::cuda::EngineConfig::getDevice;
%rename(set_device) TissueForge::cuda::EngineConfig::setDevice;
%rename(clear_device) TissueForge::cuda::EngineConfig::clearDevice;
%rename(to_device) TissueForge::cuda::EngineConfig::toDevice;
%rename(from_device) TissueForge::cuda::EngineConfig::fromDevice;
%rename(set_blocks) TissueForge::cuda::EngineConfig::setBlocks;
%rename(set_threads) TissueForge::cuda::EngineConfig::setThreads;
%rename(refresh_potentials) TissueForge::cuda::EngineConfig::refreshPotentials;
%rename(refresh_fluxes) TissueForge::cuda::EngineConfig::refreshFluxes;
%rename(refresh_boundary_conditions) TissueForge::cuda::EngineConfig::refreshBoundaryConditions;

%rename(_cuda_SimulatorConfig) TissueForge::cuda::SimulatorConfig;
%rename(_cuda_AngleConfig) TissueForge::cuda::AngleConfig;
%rename(_cuda_BondConfig) TissueForge::cuda::BondConfig;
%rename(_cuda_EngineConfig) TissueForge::cuda::EngineConfig;
%rename(_cuda_init) TissueForge::cuda::init;
%rename(_cuda_setGLDevice) TissueForge::cuda::setGLDevice;
%rename(_cuda_getDeviceName) TissueForge::cuda::getDeviceName;
%rename(_cuda_getDeviceTotalMem) TissueForge::cuda::getDeviceTotalMem;
%rename(_cuda_getDeviceAttribute) TissueForge::cuda::getDeviceAttribute;
%rename(_cuda_getNumDevices) TissueForge::cuda::getNumDevices;
%rename(_cuda_getDevicePCIBusId) TissueForge::cuda::getDevicePCIBusId;
%rename(_cuda_getCurrentDevice) TissueForge::cuda::getCurrentDevice;
%rename(_cuda_maxThreadsPerBlock) TissueForge::cuda::maxThreadsPerBlock;
%rename(_cuda_maxBlockDimX) TissueForge::cuda::maxBlockDimX;
%rename(_cuda_maxBlockDimY) TissueForge::cuda::maxBlockDimY;
%rename(_cuda_maxBlockDimZ) TissueForge::cuda::maxBlockDimZ;
%rename(_cuda_maxGridDimX) TissueForge::cuda::maxGridDimX;
%rename(_cuda_maxGridDimY) TissueForge::cuda::maxGridDimY;
%rename(_cuda_maxGridDimZ) TissueForge::cuda::maxGridDimZ;
%rename(_cuda_maxSharedMemPerBlock) TissueForge::cuda::maxSharedMemPerBlock;
%rename(_cuda_maxTotalMemConst) TissueForge::cuda::maxTotalMemConst;
%rename(_cuda_warpSize) TissueForge::cuda::warpSize;
%rename(_cuda_maxRegsPerBlock) TissueForge::cuda::maxRegsPerBlock;
%rename(_cuda_clockRate) TissueForge::cuda::clockRate;
%rename(_cuda_gpuOverlap) TissueForge::cuda::gpuOverlap;
%rename(_cuda_numMultiprocessors) TissueForge::cuda::numMultiprocessors;
%rename(_cuda_kernelExecTimeout) TissueForge::cuda::kernelExecTimeout;
%rename(_cuda_computeModeDefault) TissueForge::cuda::computeModeDefault;
%rename(_cuda_computeModeProhibited) TissueForge::cuda::computeModeProhibited;
%rename(_cuda_computeModeExclusive) TissueForge::cuda::computeModeExclusive;
%rename(_cuda_PCIDeviceId) TissueForge::cuda::PCIDeviceId;
%rename(_cuda_PCIDomainId) TissueForge::cuda::PCIDomainId;
%rename(_cuda_clockRateMem) TissueForge::cuda::clockRateMem;
%rename(_cuda_globalMemBusWidth) TissueForge::cuda::globalMemBusWidth;
%rename(_cuda_L2CacheSize) TissueForge::cuda::L2CacheSize;
%rename(_cuda_maxThreadsPerMultiprocessor) TissueForge::cuda::maxThreadsPerMultiprocessor;
%rename(_cuda_computeCapabilityMajor) TissueForge::cuda::computeCapabilityMajor;
%rename(_cuda_computeCapabilityMinor) TissueForge::cuda::computeCapabilityMinor;
%rename(_cuda_L1CacheSupportGlobal) TissueForge::cuda::L1CacheSupportGlobal;
%rename(_cuda_L1CacheSupportLocal) TissueForge::cuda::L1CacheSupportLocal;
%rename(_cuda_maxSharedMemPerMultiprocessor) TissueForge::cuda::maxSharedMemPerMultiprocessor;
%rename(_cuda_maxRegsPerMultiprocessor) TissueForge::cuda::maxRegsPerMultiprocessor;
%rename(_cuda_managedMem) TissueForge::cuda::managedMem;
%rename(_cuda_multiGPUBoard) TissueForge::cuda::multiGPUBoard;
%rename(_cuda_multiGPUBoardGroupId) TissueForge::cuda::multiGPUBoardGroupId;
%rename(_cuda_test) TissueForge::cuda::test;
%rename(_cuda_tfIncludePath) TissueForge::cuda::tfIncludePath;
%rename(_cuda_setTfIncludePath) TissueForge::cuda::setTfIncludePath;
%rename(_cuda_tfPrivateIncludePath) TissueForge::cuda::tfPrivateIncludePath;
%rename(_cuda_tfResourcePath) TissueForge::cuda::tfResourcePath;
%rename(_cuda_setTfResourcePath) TissueForge::cuda::setTfResourcePath;
%rename(_cuda_CUDAPath) TissueForge::cuda::CUDAPath;
%rename(_cuda_CUDAIncludePath) TissueForge::cuda::CUDAIncludePath;
%rename(_cuda_setCUDAIncludePath) TissueForge::cuda::setCUDAIncludePath;
%rename(_cuda_CUDAResourcePath) TissueForge::cuda::CUDAResourcePath;
%rename(_cuda_CUDAPTXObjectRelPath) TissueForge::cuda::CUDAPTXObjectRelPath;
%rename(_cuda_CUDAArchs) TissueForge::cuda::CUDAArchs;

%include "tf_cuda.h"
%include "cuda/tfAngleConfig.h"
%include "cuda/tfBondConfig.h"
%include "cuda/tfEngineConfig.h"
%include "cuda/tfSimulatorConfig.h"

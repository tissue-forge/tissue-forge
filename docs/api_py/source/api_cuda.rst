GPU-Accelerated Modules
------------------------

.. currentmodule:: tissue_forge.cuda

.. note::

    This section of the Tissue Forge API is only available in CUDA-supported installations.


.. autoclass:: SimulatorConfig

    .. autoproperty:: engine

        :type: EngineConfig

    .. autoproperty:: angles

        :type: AngleConfig

    .. autoproperty:: bonds

        :type: BondConfig


.. autoclass:: EngineConfig

    .. automethod:: onDevice

    .. automethod:: getDevice

    .. automethod:: setDevice

    .. automethod:: clearDevice

    .. automethod:: toDevice

    .. automethod:: fromDevice

    .. automethod:: setBlocks

    .. automethod:: setThreads

    .. automethod:: refreshPotentials

    .. automethod:: refreshFluxes

    .. automethod:: refreshBoundaryConditions

    .. automethod:: refresh


.. autoclass:: BondConfig

    .. automethod:: onDevice

    .. automethod:: getDevice

    .. automethod:: setDevice

    .. automethod:: toDevice

    .. automethod:: fromDevice

    .. automethod:: setBlocks

    .. automethod:: setThreads

    .. automethod:: refreshBond

    .. automethod:: refreshBonds

    .. automethod:: refresh


.. autoclass:: AngleConfig

    .. automethod:: onDevice

    .. automethod:: getDevice

    .. automethod:: toDevice

    .. automethod:: fromDevice

    .. automethod:: setBlocks

    .. automethod:: setThreads

    .. automethod:: refreshAngle

    .. automethod:: refreshAngles

    .. automethod:: refresh


.. autofunction:: init

.. autofunction:: setGLDevice

.. autofunction:: getDeviceName

.. autofunction:: getDeviceTotalMem

.. autofunction:: getDeviceAttribute

.. autofunction:: getNumDevices

.. autofunction:: getDevicePCIBusId

.. autofunction:: getCurrentDevice

.. autofunction:: maxThreadsPerBlock

.. autofunction:: maxBlockDimX

.. autofunction:: maxBlockDimY

.. autofunction:: maxBlockDimZ

.. autofunction:: maxGridDimX

.. autofunction:: maxGridDimY

.. autofunction:: maxGridDimZ

.. autofunction:: maxSharedMemPerBlock

.. autofunction:: maxTotalMemConst

.. autofunction:: warpSize

.. autofunction:: maxRegsPerBlock

.. autofunction:: clockRate

.. autofunction:: gpuOverlap

.. autofunction:: numMultiprocessors

.. autofunction:: kernelExecTimeout

.. autofunction:: computeModeDefault

.. autofunction:: computeModeProhibited

.. autofunction:: computeModeExclusive

.. autofunction:: PCIDeviceId

.. autofunction:: PCIDomainId

.. autofunction:: clockRateMem

.. autofunction:: globalMemBusWidth

.. autofunction:: L2CacheSize

.. autofunction:: maxThreadsPerMultiprocessor

.. autofunction:: computeCapabilityMajor

.. autofunction:: computeCapabilityMinor

.. autofunction:: L1CacheSupportGlobal

.. autofunction:: L1CacheSupportLocal

.. autofunction:: maxSharedMemPerMultiprocessor

.. autofunction:: maxRegsPerMultiprocessor

.. autofunction:: managedMem

.. autofunction:: multiGPUBoard

.. autofunction:: multiGPUBoardGroupId

.. autofunction:: test

.. autofunction:: tfIncludePath

.. autofunction:: setTfIncludePath

.. autofunction:: tfPrivateIncludePath

.. autofunction:: tfResourcePath

.. autofunction:: setTfResourcePath

.. autofunction:: CUDAPath

.. autofunction:: CUDAIncludePath

.. autofunction:: setCUDAIncludePath

.. autofunction:: CUDAResourcePath

.. autofunction:: CUDAPTXObjectRelPath

.. autofunction:: CUDAArchs


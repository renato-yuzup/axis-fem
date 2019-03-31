#include "GPUManager.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

namespace afc = axis::foundation::computing;

#define ERR_CHECK(x)      {cudaError_t errId = x; if (errId != CUDA_SUCCESS) throw errId;}

#define AXIS_MIN_CUDA_COMPUTE_MAJOR_VERSION     2
#define AXIS_MIN_CUDA_COMPUTE_MINOR_VERSION     0

namespace {

  /**
   * Returns if, for a given device, the capabilities required by this
   * application are present.
   *
   * @param gpuProp The GPU properties descriptor.
   *
   * @return true if it is a compatible device, false otherwise.
  **/
  inline bool IsCompatibleDevice(const cudaDeviceProp& gpuProp)
  {
    bool versionCompatible = ((gpuProp.major == AXIS_MIN_CUDA_COMPUTE_MAJOR_VERSION && 
      gpuProp.minor >= AXIS_MIN_CUDA_COMPUTE_MINOR_VERSION) ||
      (gpuProp.major > AXIS_MIN_CUDA_COMPUTE_MAJOR_VERSION));
    bool deviceAvailable = gpuProp.computeMode != cudaComputeMode::cudaComputeModeProhibited;
    return (versionCompatible && deviceAvailable);
  }
}

afc::GPUManager::GPUManager( void )
{
  gpuCount_ = 0;
  realGpuIndex_ = nullptr;
  gpuCapabilities_ = nullptr;
}

afc::GPUManager::~GPUManager( void )
{
  if (realGpuIndex_) delete [] realGpuIndex_;
  if (gpuCapabilities_) delete [] gpuCapabilities_;
}

void afc::GPUManager::Rescan( void )
{
  std::lock_guard<std::mutex> guard(mutex_);
  PopulateGPU();
  for (int i = 0; i < gpuCount_; i++)
  {
    int gpuIndex = realGpuIndex_[i];
    gpuCapabilities_[i] = ReadGPUCapability(gpuIndex);
  }
}

int afc::GPUManager::GetAvailableGPUCount( void ) const
{
  std::lock_guard<std::mutex> guard(mutex_);
  return gpuCount_;
}

afc::GPUCapability afc::GPUManager::GetGPUCapabilities( int gpuIndex ) const
{
  return gpuCapabilities_[gpuIndex];
}

void afc::GPUManager::PopulateGPU( void )
{
  int totalGpuCount = 0;
  std::vector<int> indexes;
  try
  {
    // scan for available devices
    cudaError_t errId = cudaGetDeviceCount(&totalGpuCount);
	if (errId == cudaErrorInsufficientDriver || errId == cudaErrorNoDevice)
	{ // there are no compatible GPUs in the system
		totalGpuCount = 0;
	}
    for (int i = 0; i < totalGpuCount; i++)
    {
      cudaDeviceProp gpuProp;
      ERR_CHECK(cudaGetDeviceProperties(&gpuProp, i));
      if (IsCompatibleDevice(gpuProp))
      {
        indexes.push_back(i);
      }
    }
  }
  catch (int&)
  { // an error occurred
    // TODO: warn it
  }

  // add all gathered devices
  gpuCount_ = (int)indexes.size();
  if (realGpuIndex_) delete [] realGpuIndex_;
  if (gpuCapabilities_) delete [] gpuCapabilities_;
  realGpuIndex_ = new int[gpuCount_];
  gpuCapabilities_ = new GPUCapability[gpuCount_];
  for (int i = 0; i < gpuCount_; i++)
  {
    realGpuIndex_[i] = indexes[i];
  }
}

afc::GPUCapability afc::GPUManager::ReadGPUCapability( int index )
{
  try
  {
    cudaDeviceProp gpuProp;
    ERR_CHECK(cudaGetDeviceProperties(&gpuProp, index));
    String devName;
    StringEncoding::AssignFromASCII(gpuProp.name, devName);
    afc::GPUCapability caps(index);
    caps.Name() = devName;
    caps.GlobalMemorySize() = gpuProp.totalGlobalMem;
    caps.ConstantMemorySize() = gpuProp.totalConstMem;
    caps.MaxSharedMemoryPerBlock() = gpuProp.sharedMemPerBlock;
    caps.PCIDeviceId() = gpuProp.pciDeviceID;
    caps.PCIBusId() = gpuProp.pciBusID;
    caps.PCIDomainId() = gpuProp.pciDomainID;
    caps.MaxThreadPerMultiprocessor() = gpuProp.maxThreadsPerMultiProcessor;
    caps.MaxThreadPerBlock() = gpuProp.maxThreadsPerBlock;
    caps.MaxThreadDimX() = gpuProp.maxThreadsDim[0];
    caps.MaxThreadDimY() = gpuProp.maxThreadsDim[1];
    caps.MaxThreadDimZ() = gpuProp.maxThreadsDim[2];
    caps.MaxGridDimX() = gpuProp.maxGridSize[0];
    caps.MaxGridDimY() = gpuProp.maxGridSize[1];
    caps.MaxGridDimZ() = gpuProp.maxGridSize[2];
    caps.MaxAsynchronousOperationCount() = gpuProp.asyncEngineCount;
    caps.IsWatchdogTimerEnabled() = (gpuProp.kernelExecTimeoutEnabled != 0);
    caps.IsTeslaComputeClusterEnabled() = (gpuProp.tccDriver != 0);
    caps.IsUnifiedAddressEnabled() = (gpuProp.unifiedAddressing != 0);
    caps.IsECCEnabled() = (gpuProp.ECCEnabled != 0);
    caps.IsDiscreteDevice() = (gpuProp.integrated == 0);
    caps.MaxThreadsInGPU() = (gpuProp.maxGridSize[0]*gpuProp.maxGridSize[1]*gpuProp.maxGridSize[2]*gpuProp.maxThreadsPerBlock);
    return caps;
  }
  catch (int&)
  { // an error occurred calling CUDA runtime functions
    return GPUCapability(index);
  }
}

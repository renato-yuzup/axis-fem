#pragma once
#include "AxisString.hpp"

namespace axis { namespace foundation { namespace computing {

/**
 * Describes the capabilities of a GPU device.
 */
class GPUCapability
{
public:
  GPUCapability(uint64 id);
  GPUCapability(void);
  ~GPUCapability(void);

  uint64 GetId(void) const;
  axis::String Name(void) const;
  uint64 GlobalMemorySize(void) const;
  uint64 ConstantMemorySize(void) const;
  uint64 MaxSharedMemoryPerBlock(void) const;
  int PCIDeviceId(void) const;
  int PCIBusId(void) const;
  int PCIDomainId(void) const;
  int MaxThreadPerMultiprocessor(void) const;
  int MaxThreadsInGPU(void) const;
  int MaxThreadPerBlock(void) const;
  int MaxThreadDimX(void) const;
  int MaxThreadDimY(void) const;
  int MaxThreadDimZ(void) const;
  int MaxGridDimX(void) const;
  int MaxGridDimY(void) const;
  int MaxGridDimZ(void) const;
  int MaxAsynchronousOperationCount(void) const;
  bool IsWatchdogTimerEnabled(void) const;
  bool IsTeslaComputeClusterEnabled(void) const;
  bool IsUnifiedAddressEnabled(void) const;
  bool IsECCEnabled(void) const;
  bool IsDiscreteDevice(void) const;
  double ProcessingScore(void) const;
  double MemoryScore(void) const;
  double GridScore(void) const;

  axis::String& Name(void);
  uint64& GlobalMemorySize(void);
  uint64& ConstantMemorySize(void);
  uint64& MaxSharedMemoryPerBlock(void);
  int& PCIDeviceId(void);
  int& PCIBusId(void);
  int& PCIDomainId(void);
  int& MaxThreadPerMultiprocessor(void);
  int& MaxThreadsInGPU(void);
  int& MaxThreadPerBlock(void);
  int& MaxThreadDimX(void);
  int& MaxThreadDimY(void);
  int& MaxThreadDimZ(void);
  int& MaxGridDimX(void);
  int& MaxGridDimY(void);
  int& MaxGridDimZ(void);
  int& MaxAsynchronousOperationCount(void);
  bool& IsWatchdogTimerEnabled(void);
  bool& IsTeslaComputeClusterEnabled(void);
  bool& IsUnifiedAddressEnabled(void);
  bool& IsECCEnabled(void);
  bool& IsDiscreteDevice(void);
  double& ProcessingScore(void);
  double& MemoryScore(void);
  double& GridScore(void);
private:
  void InitToDefaultValues(void);

  uint64 id_;
  axis::String name_;
  uint64 globalMemSize_;
  uint64 constMemSize_;
  uint64 sharedMemPerBlock_;
  int pciId_;
  int busId_;
  int domainId_;
  int maxThreadPerMultiproc_;
  int maxThreadNumInGpu_;
  int maxThreadPerBlock_;
  int maxThreadDimX_;
  int maxThreadDimY_;
  int maxThreadDimZ_;
  int maxGridDimX_;
  int maxGridDimY_;
  int maxGridDimZ_;
  int maxAsyncOpCount_;
  bool wdtEnabled_;
  bool tccEnabled_;
  bool uaEnabled_;
  bool eccEnabled_;
  bool isDiscrete_;
  double procesingScore_;
  double memoryScore_;
  double gridScore_;
};

} } } // namespace axis::foundation::computing

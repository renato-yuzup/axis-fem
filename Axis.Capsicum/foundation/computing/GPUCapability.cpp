#include "GPUCapability.hpp"

namespace afc = axis::foundation::computing;

afc::GPUCapability::GPUCapability( void )
{
  id_ = 0;
  InitToDefaultValues();
}

afc::GPUCapability::GPUCapability( uint64 id )
{
  id_ = id;
  InitToDefaultValues();
}

afc::GPUCapability::~GPUCapability( void )
{
  // nothing to do here
}

uint64 afc::GPUCapability::GetId( void ) const
{
  return id_;
}

axis::String afc::GPUCapability::Name( void ) const
{
  return name_;
}

axis::String& afc::GPUCapability::Name( void )
{
  return name_;
}

uint64 afc::GPUCapability::GlobalMemorySize( void ) const
{
  return globalMemSize_;
}

uint64& afc::GPUCapability::GlobalMemorySize( void )
{
  return globalMemSize_;
}

uint64 afc::GPUCapability::ConstantMemorySize( void ) const
{
  return constMemSize_;
}

uint64& afc::GPUCapability::ConstantMemorySize( void )
{
  return constMemSize_;
}

uint64 afc::GPUCapability::MaxSharedMemoryPerBlock( void ) const
{
  return sharedMemPerBlock_;
}

uint64& afc::GPUCapability::MaxSharedMemoryPerBlock( void )
{
  return sharedMemPerBlock_;
}

int afc::GPUCapability::PCIDeviceId( void ) const
{
  return pciId_;
}

int& afc::GPUCapability::PCIDeviceId( void )
{
  return pciId_;
}

int afc::GPUCapability::PCIBusId( void ) const
{
  return busId_;
}

int& afc::GPUCapability::PCIBusId( void )
{
  return busId_;
}

int afc::GPUCapability::PCIDomainId( void ) const
{
  return domainId_;
}

int& afc::GPUCapability::PCIDomainId( void )
{
  return domainId_;
}

int afc::GPUCapability::MaxThreadPerMultiprocessor( void ) const
{
  return maxThreadPerMultiproc_;
}

int& afc::GPUCapability::MaxThreadPerMultiprocessor( void )
{
  return maxThreadPerMultiproc_;
}

int afc::GPUCapability::MaxThreadsInGPU( void ) const
{
  return maxThreadNumInGpu_;
}

int& afc::GPUCapability::MaxThreadsInGPU( void )
{
  return maxThreadNumInGpu_;
}

int afc::GPUCapability::MaxThreadPerBlock( void ) const
{
  return maxThreadPerBlock_;
}

int& afc::GPUCapability::MaxThreadPerBlock( void )
{
  return maxThreadPerBlock_;
}

int afc::GPUCapability::MaxThreadDimX( void ) const
{
  return maxThreadDimX_;
}

int& afc::GPUCapability::MaxThreadDimX( void )
{
  return maxThreadDimX_;
}

int afc::GPUCapability::MaxThreadDimY( void ) const
{
  return maxThreadDimY_;
}

int& afc::GPUCapability::MaxThreadDimY( void )
{
  return maxThreadDimY_;
}

int afc::GPUCapability::MaxThreadDimZ( void ) const
{
  return maxThreadDimZ_;
}

int& afc::GPUCapability::MaxThreadDimZ( void )
{
  return maxThreadDimZ_;
}

int afc::GPUCapability::MaxGridDimX( void ) const
{
  return maxGridDimX_;
}

int& afc::GPUCapability::MaxGridDimX( void )
{
  return maxGridDimX_;
}

int afc::GPUCapability::MaxGridDimY( void ) const
{
  return maxGridDimY_;
}

int& afc::GPUCapability::MaxGridDimY( void )
{
  return maxGridDimY_;
}

int afc::GPUCapability::MaxGridDimZ( void ) const
{
  return maxGridDimZ_;
}

int& afc::GPUCapability::MaxGridDimZ( void )
{
  return maxGridDimZ_;
}

int afc::GPUCapability::MaxAsynchronousOperationCount( void ) const
{
  return maxAsyncOpCount_;
}

int& afc::GPUCapability::MaxAsynchronousOperationCount( void )
{
  return maxAsyncOpCount_;
}

bool afc::GPUCapability::IsWatchdogTimerEnabled( void ) const
{
  return wdtEnabled_;
}

bool& afc::GPUCapability::IsWatchdogTimerEnabled( void )
{
  return wdtEnabled_;
}

bool afc::GPUCapability::IsTeslaComputeClusterEnabled( void ) const
{
  return tccEnabled_;
}

bool& afc::GPUCapability::IsTeslaComputeClusterEnabled( void )
{
  return tccEnabled_;
}

bool afc::GPUCapability::IsUnifiedAddressEnabled( void ) const
{
  return uaEnabled_;
}

bool& afc::GPUCapability::IsUnifiedAddressEnabled( void )
{
  return uaEnabled_;
}

bool afc::GPUCapability::IsECCEnabled( void ) const
{
  return eccEnabled_;
}

bool& afc::GPUCapability::IsECCEnabled( void )
{
  return eccEnabled_;
}

bool afc::GPUCapability::IsDiscreteDevice( void ) const
{
  return isDiscrete_;
}

bool& afc::GPUCapability::IsDiscreteDevice( void )
{
  return isDiscrete_;
}

double afc::GPUCapability::ProcessingScore( void ) const
{
  return procesingScore_;
}

double& afc::GPUCapability::ProcessingScore( void )
{
  return procesingScore_;
}

double afc::GPUCapability::MemoryScore( void ) const
{
  return memoryScore_;
}

double& afc::GPUCapability::MemoryScore( void )
{
  return memoryScore_;
}

double afc::GPUCapability::GridScore( void ) const
{
  return gridScore_;
}

double& afc::GPUCapability::GridScore( void )
{
  return gridScore_;
}

void afc::GPUCapability::InitToDefaultValues( void )
{
  globalMemSize_ = 0; constMemSize_ = 0; sharedMemPerBlock_ = 0;
  pciId_ = -1; busId_ = -1; domainId_ = -1;
  maxThreadPerBlock_ = 0; maxThreadPerMultiproc_ = 0;
  maxThreadDimX_ = 0; maxThreadDimY_ = 0; maxThreadDimZ_ = 0;
  maxGridDimX_ = 0; maxGridDimY_ = 0; maxGridDimZ_ = 0;
  maxAsyncOpCount_ = 0;
  wdtEnabled_ = false;
  tccEnabled_ = false;
  uaEnabled_ = false;
  eccEnabled_ = false;
  isDiscrete_ = false;
  procesingScore_ = 0;
  memoryScore_ = 0;
  gridScore_ = 0;
}

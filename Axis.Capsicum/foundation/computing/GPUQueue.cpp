#include "GPUQueue.hpp"
#include <cuda.h>
#include <iostream>
#include "AxisString.hpp"
#include "foundation/ApplicationErrorException.hpp"

namespace afc = axis::foundation::computing;

afc::GPUQueue::GPUQueue(int deviceIndex)
{
  deviceIndex_ = deviceIndex;
  deviceStream_ = nullptr;
  gpuEvent_ = nullptr;
}

afc::GPUQueue::~GPUQueue(void)
{
  if (deviceStream_ != nullptr)
  {
    cudaEventDestroy(gpuEvent_);
    cudaStreamDestroy(deviceStream_);
  }
}

int afc::GPUQueue::GetQueueIndex( void ) const
{
  return deviceIndex_;
}

cudaStream_t afc::GPUQueue::GetQueue( void )
{
  LazyInstantiate();
  return deviceStream_;
}

const cudaStream_t afc::GPUQueue::GetQueue( void ) const
{
  LazyInstantiate();
  return deviceStream_;
}

cudaEvent_t afc::GPUQueue::GetEvent( void )
{
  LazyInstantiate();
  return gpuEvent_;
}

const cudaEvent_t afc::GPUQueue::GetEvent( void ) const
{
  LazyInstantiate();
  return gpuEvent_;
}

void afc::GPUQueue::Synchronize( void )
{
  cudaEventSynchronize(gpuEvent_);
  // cudaStreamSynchronize(deviceStream_);
}

void afc::GPUQueue::LazyInstantiate( void ) const
{
  if (deviceStream_ == nullptr)
  { // lazy instantiate stream object
    try
    {
      if (cudaSetDevice(deviceIndex_) != CUDA_SUCCESS) throw 1;
      if (cudaStreamCreate(&deviceStream_) != CUDA_SUCCESS) throw 1;
      if (cudaEventCreateWithFlags(&gpuEvent_, cudaEventDisableTiming) != CUDA_SUCCESS) throw 1;
    }
    catch (int)
    { // CUDA function failed!
      cudaError_t errId = cudaGetLastError();
      const char * err = cudaGetErrorString(errId);
      String errStr;
      StringEncoding::AssignFromASCII(err, errStr);
      errStr += _T(" (") + String::int_parse((int)errId) + _T(")");
      throw axis::foundation::ApplicationErrorException(errStr);
    }
  }
}

void afc::GPUQueue::RequestSynchronization( void )
{
  LazyInstantiate();
  cudaEventRecord(gpuEvent_, deviceStream_);
}
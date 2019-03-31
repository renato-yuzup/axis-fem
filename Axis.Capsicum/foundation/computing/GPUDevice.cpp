#include "GPUDevice.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "GPUQueue.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "yuzu/common/gpu.hpp"

namespace afc = axis::foundation::computing;

#if defined(_DEBUG) || defined(DEBUG)
  #define ASSERT_SUCCESS(x)   \
    { cudaError_t err = x; \
      if (err != CUDA_SUCCESS) \
        throw axis::foundation::ApplicationErrorException(); \
    }
#else
  #define ASSERT_SUCCESS(x) x
#endif

afc::GPUDevice::GPUDevice( const GPUCapability& caps, uint64 totalMemorySize, 
  size_type maxThreadCount, GPUQueue& queue ) :
capabilities_(caps)
{
  queue_ = &queue;
  memorySize_ = totalMemorySize;
  maxThreadCount_ = maxThreadCount;
  initialized_ = false;
}

afc::GPUDevice::~GPUDevice( void )
{
  // nothing to do here
}

afc::GPUQueue& afc::GPUDevice::GetQueue( void )
{
  return *queue_;
}

uint64 afc::GPUDevice::GetTotalMemorySize( void ) const
{
  return memorySize_;
}

size_type afc::GPUDevice::GetMaxThreadCount( void ) const
{
  return maxThreadCount_;
}

size_type afc::GPUDevice::GetMaxThreadPerBlock( void ) const
{
  return AXIS_YUZU_MAX_THREADS_PER_BLOCK;
}

int afc::GPUDevice::GetMaxThreadDimX( void ) const
{
  return capabilities_.MaxThreadDimX();
}

int afc::GPUDevice::GetMaxThreadDimY( void ) const
{
  return capabilities_.MaxThreadDimY();
}

int afc::GPUDevice::GetMaxThreadDimZ( void ) const
{
  return capabilities_.MaxThreadDimZ();
}

int afc::GPUDevice::GetMaxGridDimX( void ) const
{
  return capabilities_.MaxGridDimX();
}

int afc::GPUDevice::GetMaxGridDimY( void ) const
{
  return capabilities_.MaxGridDimY();
}

int afc::GPUDevice::GetMaxGridDimZ( void ) const
{
  return capabilities_.MaxGridDimZ();
}

void afc::GPUDevice::SetActive( void )
{
  int devIndex = queue_->GetQueueIndex();
  ASSERT_SUCCESS(cudaSetDevice(devIndex));
  if (!initialized_)
  {
    ASSERT_SUCCESS(cudaDeviceReset());
    ASSERT_SUCCESS(cudaSetDeviceFlags(cudaDeviceMapHost | 
      cudaDeviceScheduleAuto | cudaDeviceScheduleBlockingSync));
    initialized_ = true;
  }
}

void * afc::GPUDevice::AllocateMemory( uint64 bytes )
{
  SetActive();
  void *ptr = nullptr;
  ASSERT_SUCCESS(cudaMalloc(&ptr, bytes));
  if (ptr == nullptr)
  {
    throw axis::foundation::OutOfMemoryException();
  }
  return ptr;
}

void afc::GPUDevice::DeallocateMemory( void *memAddr )
{
  SetActive();
  ASSERT_SUCCESS(cudaFree(memAddr));
}

void afc::GPUDevice::Synchronize( void )
{
  queue_->Synchronize();
}

void afc::GPUDevice::SendToDevice( void *targetDeviceMemAddress, 
  const void *srcHostMemAddress, uint64 blockSize )
{
  ASSERT_SUCCESS(cudaMemcpy(targetDeviceMemAddress, srcHostMemAddress, 
                            blockSize, ::cudaMemcpyHostToDevice));
}

void afc::GPUDevice::ReadFromDevice( void *targetHostMemAddress, 
  const void *srcDeviceMemAddress, uint64 blockSize )
{
  ASSERT_SUCCESS(cudaMemcpy(targetHostMemAddress, srcDeviceMemAddress, 
                            blockSize, ::cudaMemcpyDeviceToHost));
}

void afc::GPUDevice::AttachMemory( void *targetMemory, uint64 blockSize )
{
  SetActive();
  ASSERT_SUCCESS(cudaHostRegister(targetMemory, blockSize, 
    cudaHostRegisterPortable | cudaHostRegisterMapped));
}

void afc::GPUDevice::DetachMemory( void *targetMemory )
{
  SetActive();
  ASSERT_SUCCESS(cudaHostUnregister(targetMemory));
}

void * afc::GPUDevice::GetAttachedMemoryAddress( void *attachedHostMemAddr )
{
  SetActive();
  void *devPtr;
  ASSERT_SUCCESS(cudaHostGetDevicePointer(&devPtr, attachedHostMemAddr, 0));
  return devPtr;
}

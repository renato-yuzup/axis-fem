#include "KernelConfiguration.hpp"
#include <cmath>
#include "foundation/computing/GPUQueue.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "foundation/ApplicationErrorException.hpp"

#define min(x,y) (x > y? y : x)

namespace assg = axis::services::scheduling::gpu;
namespace afc  = axis::foundation::computing;
namespace asmc = axis::services::memory::commands;
namespace afm  = axis::foundation::memory;
namespace afmo = axis::foundation::mirroring;

#if defined(DEBUG) || defined(_DEBUG)
#define CUDA_ASSERT(x)  {cudaError_t err = x; if (err != CUDA_SUCCESS) \
                         throw 0xDEAD; }
#else
#define CUDA_ASSERT(x)  x;
#endif

namespace {
void MinimizeCube(axis::Dimension3D& cube, size_type elementsToAllocate,
                  size_type maxX, size_type maxY, size_type maxZ)
{
  size_type dimXExp = (size_type)floor(log10(maxX) / log10(2.0));
  size_type dimYExp = (size_type)floor(log10(maxY) / log10(2.0));
  size_type dimZExp = (size_type)floor(log10(maxZ) / log10(2.0));
  size_type elementsExp = 
    (size_type)ceil(log10(elementsToAllocate) / log10(2.0));
  elementsExp -= (cube.X = min(dimXExp, elementsExp));
  elementsExp -= (cube.Y = min(dimYExp, elementsExp));
  elementsExp -= (cube.Z = min(dimZExp, elementsExp));
  cube.X = (unsigned int)pow(2, cube.X);
  cube.Y = (unsigned int)pow(2, cube.Y);
  cube.Z = (unsigned int)pow(2, cube.Z);
}

void MinimizeConfigurationGrid(axis::Dimension3D& grid, axis::Dimension3D& block, 
  size_type elementsToAllocate, size_type maxElementsInBlock,
  size_type maxBlockX, size_type maxBlockY, size_type maxBlockZ,
  size_type maxGridX, size_type maxGridY, size_type maxGridZ)
{
  size_type elementsInBlock = elementsToAllocate > maxElementsInBlock? 
                              maxElementsInBlock : elementsToAllocate;
  size_type blocksNum = 
    (size_type)ceil((double)elementsToAllocate / (double)maxElementsInBlock);
  MinimizeCube(block, elementsInBlock, maxBlockX, maxBlockY, maxBlockZ);
  MinimizeCube(grid, blocksNum, maxGridX, maxGridY, maxGridZ);
}
} // namespace

assg::KernelConfiguration::KernelConfiguration(afc::GPUDevice& gpu, 
  afmo::MemoryMirror& memMirror, asmc::MemoryCommand& memCommand, 
  size_type elementCount) : gpu_(&gpu), memMirror_(&memMirror), 
  memCommand_(&memCommand)
{
  elementCount_ = elementCount;  
}

assg::KernelConfiguration::~KernelConfiguration(void)
{
  delete memMirror_;
  delete memCommand_;
}

void assg::KernelConfiguration::Allocate( void )
{
  memMirror_->Allocate();
}

void assg::KernelConfiguration::Deallocate( void )
{
  memMirror_->Deallocate();
}

void assg::KernelConfiguration::Mirror( void )
{
  memMirror_->Mirror();
}

void assg::KernelConfiguration::Restore( void )
{
  memMirror_->Restore();
}

void assg::KernelConfiguration::InitMemory( void )
{
  gpu_->SetActive();
  void *cpuBaseAddress = memMirror_->GetHostBaseAddress();
  void *gpuBaseAddress = memMirror_->GetGPUBaseAddress();
  memCommand_->Execute(cpuBaseAddress, gpuBaseAddress);
}

void assg::KernelConfiguration::ClaimMemory( void *baseAddress, 
  uint64 blockSize )
{
  gpu_->AttachMemory(baseAddress, blockSize);
}

void assg::KernelConfiguration::ReturnMemory( void *baseAddress )
{
  gpu_->DetachMemory(baseAddress);
}

void assg::KernelConfiguration::RunCommand( afc::KernelCommand& command, 
  size_type baseIndex )
{
  auto& queue = gpu_->GetQueue();
  size_type maxThreads = gpu_->GetMaxThreadPerBlock();  
  size_type gridDimX =  gpu_->GetMaxGridDimX();
  size_type gridDimY =  gpu_->GetMaxGridDimY();
  size_type gridDimZ =  gpu_->GetMaxGridDimZ();
  size_type blockDimX =  gpu_->GetMaxThreadDimX();
  size_type blockDimY =  gpu_->GetMaxThreadDimY();
  size_type blockDimZ =  gpu_->GetMaxThreadDimZ();
  Dimension3D grid, block;
  MinimizeConfigurationGrid(grid, block, elementCount_, maxThreads, blockDimX, 
    blockDimY, blockDimZ, gridDimX, gridDimY, gridDimZ);  
  cudaGetLastError();
  gpu_->SetActive();
  void *streamPtr = queue.GetQueue();
  cudaError_t errId = cudaGetLastError();
  if (errId != CUDA_SUCCESS)
  {
    const char *s = cudaGetErrorString(errId);
    String errStr;
    StringEncoding::AssignFromASCII(s, errStr);
    errStr = String::int_parse((long)errId) + _T(": ") + errStr;
    throw axis::foundation::ApplicationErrorException(errStr);
  }
  command.Run(elementCount_, baseIndex, memMirror_->GetGPUBaseAddress(), 
    grid, block, streamPtr);

  // append synchronization to device stream
  queue.RequestSynchronization();
}

size_type assg::KernelConfiguration::GetElementCount( void ) const
{
  return elementCount_;
}

void assg::KernelConfiguration::Synchronize( void )
{
  cudaError_t errId = cudaGetLastError();
  if (errId != CUDA_SUCCESS)
  {
    const char *s = cudaGetErrorString(errId);
    String errStr;
    StringEncoding::AssignFromASCII(s, errStr);
    std::wcout << errStr.c_str() << std::endl;
    throw axis::foundation::ApplicationErrorException(errStr);
  }
  gpu_->Synchronize();
  errId = cudaGetLastError();
  if (errId != CUDA_SUCCESS)
  {
    const char *s = cudaGetErrorString(errId);
    String errStr;
    StringEncoding::AssignFromASCII(s, errStr);
    errStr = String::int_parse((long)errId) + _T(": ") + errStr;
    std::wcout << errStr.c_str() << std::endl;
    throw axis::foundation::ApplicationErrorException(errStr);
  }
}

void * assg::KernelConfiguration::GetDeviceMemoryAddress( void ) const
{
  return memMirror_->GetGPUBaseAddress();
}

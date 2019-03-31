#include "GPUBlockMirror.hpp"
#include <cstdlib>
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace afmg = axis::foundation::mirroring::gpu;
namespace afc = axis::foundation::computing;

afmg::GPUBlockMirror::GPUBlockMirror(uint64 blockSize, 
  afc::GPUDevice& targetDevice) : targetDevice_(targetDevice)
{
  if (blockSize == 0)
  {
    throw axis::foundation::ArgumentException();
  }
  allocated_ = false;
  blockSize_ = blockSize;
  deviceBlockAddress_ = nullptr;
  hostBlockAddress_ = malloc(blockSize);
  if (hostBlockAddress_ == nullptr)
  {
    throw axis::foundation::OutOfMemoryException();
  }
}

afmg::GPUBlockMirror::~GPUBlockMirror(void)
{
  delete hostBlockAddress_;
  hostBlockAddress_ = nullptr;
}

void * afmg::GPUBlockMirror::GetHostBaseAddress( void ) const
{
  return hostBlockAddress_;
}

void * afmg::GPUBlockMirror::GetGPUBaseAddress( void ) const
{
  return deviceBlockAddress_;
}

uint64 afmg::GPUBlockMirror::GetBlockSize( void ) const
{
  return blockSize_;
}

void afmg::GPUBlockMirror::Allocate( void )
{
  if (allocated_)
  { // cannot re-allocate
    throw axis::foundation::InvalidOperationException();
  }
  deviceBlockAddress_ = targetDevice_.AllocateMemory(blockSize_);
  allocated_ = true;
}

void afmg::GPUBlockMirror::Mirror( void )
{
  if (!allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  targetDevice_.SendToDevice(deviceBlockAddress_, hostBlockAddress_, 
    blockSize_);
}

void afmg::GPUBlockMirror::Restore( void )
{
  if (!allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  targetDevice_.ReadFromDevice(hostBlockAddress_, deviceBlockAddress_, 
    blockSize_);
}

void afmg::GPUBlockMirror::Deallocate( void )
{
  if (!allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  targetDevice_.DeallocateMemory(deviceBlockAddress_);
  allocated_ = false;
  deviceBlockAddress_ = nullptr;
}

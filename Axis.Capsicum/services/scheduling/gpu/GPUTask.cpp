#include "GPUTask.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "KernelConfiguration.hpp"
#include <iostream>

namespace assg = axis::services::scheduling::gpu;
namespace afc = axis::foundation::computing;

assg::GPUTask::GPUTask(void)
{
  // nothing to do here
}

assg::GPUTask::~GPUTask(void)
{
  size_type count = (size_type)kernels_.size();
  for (size_type i = 0; i < count; ++i)
  {
    delete kernels_[i];
  }
  kernels_.clear();
}

void assg::GPUTask::AddKernel( KernelConfiguration& kernelConfig )
{
  kernels_.push_back(&kernelConfig);
}

void assg::GPUTask::AllocateMemory( void )
{
  size_type count = (size_type)kernels_.size();
  for (size_type i = 0; i < count; ++i)
  {
    kernels_[i]->Allocate();
  }
}

void assg::GPUTask::DeallocateMemory( void )
{
  size_type count = (size_type)kernels_.size();
  for (size_type i = 0; i < count; ++i)
  {
    kernels_[i]->Deallocate();
  }
}

void assg::GPUTask::Mirror( void )
{
  size_type count = (size_type)kernels_.size();
  for (size_type i = 0; i < count; ++i)
  {
    kernels_[i]->Mirror();
  }
}

void assg::GPUTask::Restore( void )
{
  size_type count = (size_type)kernels_.size();
  for (size_type i = 0; i < count; ++i)
  {
    kernels_[i]->Restore();
  }
}

void assg::GPUTask::InitMemory( void )
{
  size_type count = (size_type)kernels_.size();
  for (size_type i = 0; i < count; ++i)
  {
    kernels_[i]->InitMemory();
  }
}

void assg::GPUTask::ClaimMemory( void *baseAddress, uint64 blockSize )
{
  // claiming memory to device is a global task, so requesting it to a single 
  // kernel config should suffice
  if (!kernels_.empty())
  {
    kernels_[0]->ClaimMemory(baseAddress, blockSize);
  }
}

void assg::GPUTask::ReturnMemory( void *baseAddress )
{
  if (!kernels_.empty())
  {
    kernels_[0]->ReturnMemory(baseAddress);
  }
}

void assg::GPUTask::RunCommand( afc::KernelCommand& command )
{
  size_type count = (size_type)kernels_.size();
  size_type startIndex = 0;
  for (size_type i = 0; i < count; ++i)
  {
    KernelConfiguration& kernelConfig = *kernels_[i];
    kernelConfig.RunCommand(command, startIndex);
    startIndex += kernelConfig.GetElementCount();
  }
}

void assg::GPUTask::Synchronize( void )
{
  size_type count = (size_type)kernels_.size();
  size_type startIndex = 0;
  for (size_type i = 0; i < count; ++i)
  {
    KernelConfiguration& kernelConfig = *kernels_[i];
    kernelConfig.Synchronize();
  }
}

void * assg::GPUTask::GetDeviceMemoryAddress( int index ) const
{
  return kernels_[index]->GetDeviceMemoryAddress();
}

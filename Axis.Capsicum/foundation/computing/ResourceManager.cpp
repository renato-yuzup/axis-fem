#include "ResourceManager.hpp"
#include "GPUDevice.hpp"
#include "GPUQueue.hpp"
#include "foundation/ArgumentException.hpp"

namespace afc = axis::foundation::computing;

afc::ResourceManager::ResourceManager( void )
{
  manager_.Rescan();
}

afc::ResourceManager::~ResourceManager( void )
{
  // nothing to do here
}

bool afc::ResourceManager::IsCPUResourcesAvailable( void ) const
{
  return true;
}

bool afc::ResourceManager::IsGPUResourcesAvailable( void ) const
{
  return !gpus_.empty();
}

afc::GPUDevice * afc::ResourceManager::AllocateGPU( void )
{
  if (freeGpus_.empty()) return nullptr;
  GPUDevice *dev = freeGpus_.back();
  freeGpus_.pop_back();
  return dev;
}

bool afc::ResourceManager::ReleaseGPU( GPUDevice& device )
{
  size_type gpuCount = (size_type)gpus_.size();
  for (size_type i = 0; i < gpuCount; ++i)
  {
    if (&device == gpus_[i])
    { // ok to add back to free list
      freeGpus_.push_back(&device);
      return true;
    }
  }

  // couldn't find GPU
  return false;
}

void afc::ResourceManager::InitGPUList( void )
{
  int gpuCount = manager_.GetAvailableGPUCount();
  gpus_.clear();
  freeGpus_.clear();
  for (int i = 0; i < gpuCount; ++i)
  {
    GPUCapability caps = manager_.GetGPUCapabilities(i);
    GPUQueue *queue = new GPUQueue(i);

    // we will suppose some of this memory is already used
    uint64 gpuMemSize = caps.GlobalMemorySize();
    gpuMemSize = (uint64)(gpuMemSize*0.9);

    GPUDevice *dev = new GPUDevice(caps, gpuMemSize, caps.MaxThreadsInGPU(), *queue);
    gpus_.push_back(dev);
    freeGpus_.push_back(dev);
  }
}

void afc::ResourceManager::Rescan( void )
{
  InitGPUList();
}

int afc::ResourceManager::GetAvailableGPUCount(void) const
{
	return (int)gpus_.size();
}

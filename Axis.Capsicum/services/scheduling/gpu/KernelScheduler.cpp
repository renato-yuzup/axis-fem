#include "KernelScheduler.hpp"
#include "GPUTask.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "foundation/capsicum_error.hpp"
#include "foundation/computing/GPUDevice.hpp"
#include "foundation/computing/GPUQueue.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "foundation/capsicum_warning.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "System.hpp"
#include "foundation/mirroring/gpu/GPUBlockMirror.hpp"
#include "foundation/mirroring/gpu/GPUHeapMirror.hpp"
#include "services/memory/MemoryGrid.hpp"
#include "foundation/mirroring/NullMemory.hpp"

namespace asmc = axis::services::memory::commands;
namespace assg = axis::services::scheduling::gpu;
namespace afc  = axis::foundation::computing;
namespace asmm = axis::services::messaging;
namespace afm  = axis::foundation::memory;
namespace afmo = axis::foundation::mirroring;
namespace afmg = axis::foundation::mirroring::gpu;

class assg::KernelScheduler::GPUMetadata
{
public:
  afc::GPUDevice *Device;
  bool Used;
  uint64 UsedMemorySpace;

  GPUMetadata(void)
  {
    UsedMemorySpace = 0;
    Used = false;
    Device = nullptr;
  }
};

assg::KernelScheduler::KernelScheduler(afc::ResourceManager& resourceManager) :
resourceManager_(resourceManager)
{
 globalMemAllocated_ = false;
}

assg::KernelScheduler::~KernelScheduler(void)
{
  size_type count = (size_type)allocatedGPUs_.size();
  for (size_type i = 0; i < count; ++i)
  {
    delete allocatedGPUs_[i];
  }
  allocatedGPUs_.clear();
}

void assg::KernelScheduler::BeginGPURound( void )
{
  curRoundAllocatedDevs_.clear();
}

void assg::KernelScheduler::EndGPURound( void )
{
  // mark again GPU as not used
  size_type count = (size_type)allocatedGPUs_.size();
  for (size_type i = 0; i < count; ++i)
  {
    allocatedGPUs_[i]->Used = false;
  }
}

assg::GPUTask *assg::KernelScheduler::ScheduleLocal( 
  const asmc::ElementInitializerCommand& memInitializerCmd,
  const axis::services::memory::MemoryLayout& blockLayout,
  uint64 elementCount, GPUSink& errorSink)
{
  GPUTask *task = new GPUTask();
  size_type blockSize = blockLayout.GetSegmentSize();
  uint64 remainingElements = elementCount;
  uint64 relativeIndex = 0;
  while (remainingElements > 0)
  {
    // acquire a GPU to allocate work to
    GPUMetadata *metadata = Allocate();
    if (metadata == nullptr)
    { // not enough resources
      delete task;
      errorSink.Notify(asmm::ErrorMessage(
        AXIS_ERROR_ID_SCHEDULING_UNAVAILABLE_RESOURCE,
        AXIS_ERROR_MSG_SCHEDULING_UNAVAILABLE_RESOURCE));
      return nullptr;
    }
    // check GPU caps and split work if needed
    afc::GPUDevice *gpu = metadata->Device;
    metadata->Used = true;
    uint64 gpuAvailableSpace = 
      gpu->GetTotalMemorySize() - metadata->UsedMemorySpace;    
    uint64 elementsToAllocate = remainingElements;
    if (gpuAvailableSpace < elementCount*blockSize)
    {
      elementsToAllocate = (size_type)(gpuAvailableSpace / blockSize);
    }
    if (elementsToAllocate > gpu->GetMaxThreadCount())
    {
      elementsToAllocate = gpu->GetMaxThreadCount();
    }
    metadata->UsedMemorySpace += elementsToAllocate*blockSize;
    asmc::ElementInitializerCommand &cmd = memInitializerCmd.Slice(
      relativeIndex, elementsToAllocate);
    remainingElements -=elementsToAllocate;
    relativeIndex += elementsToAllocate;

    // create mirrored memory and set up memory command
    uint64 memSize = blockSize*elementsToAllocate;
    afmg::GPUBlockMirror &mirroredMem = 
      *new afmg::GPUBlockMirror(memSize, *metadata->Device);
    cmd.SetTargetGridLayout(blockLayout, elementsToAllocate);

    // create kernel configuration
    KernelConfiguration *config = CreateKernelConfig(*metadata->Device, 
      mirroredMem, cmd, elementsToAllocate);
    if (config == nullptr)
    { // kernel configuration failed
      delete task;
      errorSink.Notify(asmm::ErrorMessage(AXIS_ERROR_ID_SCHEDULING_FATAL_ERROR,
        AXIS_ERROR_MSG_SCHEDULING_FATAL_ERROR));
      return nullptr;
    }
    task->AddKernel(*config);
  } 

  return task;
}

assg::KernelScheduler::GPUMetadata *assg::KernelScheduler::Allocate( void )
{
  /*
   * GPU ALLOCATION POLICY:
   * The algorithm implemented below used a greedy approach that seeks best 
   * performance. Although not suitable for all environments, it is reasonable 
   * for research purposes. Another implementation that focus on other aspects 
   * can replace the code in this method, changing the behavior of how the 
   * scheduler selects resources.
   * 
   * SUGGESTION: The Strategy pattern could be used here to let vary selection 
   * procedures and/or let the strategy be adaptive to environment constraints.
  **/

  GPUMetadata *metadata = nullptr;
  if (!allocatedGPUs_.empty())
  {  // try to use an existing and not used GPU before requesting a new one
    metadata = GetUnusedGPU();

    // Try to peek a new one if everything is being used
    if (metadata == nullptr) metadata = AllocateNewGPU();

    // If we couldn't get a new GPU, fallback to the least used GPU
    if (metadata == nullptr) metadata = GetLeastUsedGPU();

    // If even after that we can't get a serviceable GPU, fail
    return metadata; // failing <= (metadata == nullptr)
  }

  // No GPU has been allocated, reserve a new GPU for our use
  metadata = AllocateNewGPU();
  return metadata;
}

assg::KernelScheduler::GPUMetadata * assg::KernelScheduler::AllocateNewGPU(void)
{
  // reserve a new GPU for our use
  afc::GPUDevice *gpu = resourceManager_.AllocateGPU();
  if (gpu == nullptr) // couldn't allocate a new GPU, fail
  {
    return nullptr;
  }
  GPUMetadata *metadata = new GPUMetadata();
  metadata->Device = gpu;
  allocatedGPUs_.push_back(metadata);  // add to GPU list
  curRoundAllocatedDevs_.push_back(metadata);
  return metadata;
}

assg::KernelScheduler::GPUMetadata * assg::KernelScheduler::GetUnusedGPU( void )
{
  size_type count = (size_type)allocatedGPUs_.size();
  for (size_type i = 0; i < count; ++i)
  {
    GPUMetadata *metadata = allocatedGPUs_[i];
    if (!metadata->Used) return metadata;
  }
  return nullptr;
}

assg::KernelScheduler::GPUMetadata * assg::KernelScheduler::GetLeastUsedGPU(void)
{
  /*
   * HOW WE SELECT THE LEAST USED GPU?
   * For simplicity, we select the GPU with the largest amount of memory 
   * available. By doing this, we also ensure that, if the selected GPU cannot 
   * be used to store data, no other will either.
  **/
  size_type count = (size_type)allocatedGPUs_.size();
  GPUMetadata *leastUsedGPU = allocatedGPUs_[0];
  for (size_type i = 1; i < count; ++i)
  {
    GPUMetadata *metadata = allocatedGPUs_[i];
    uint64 leastUsedFreeMem = leastUsedGPU->Device->GetTotalMemorySize() - 
      leastUsedGPU->UsedMemorySpace;
    uint64 curMemSize = metadata->Device->GetTotalMemorySize() - 
      metadata->UsedMemorySpace;
    if (curMemSize > leastUsedFreeMem)
    {
      leastUsedGPU = metadata;
    }
  }
  return leastUsedGPU;
}

void assg::KernelScheduler::ReleaseDevices( GPUSink& errorSink )
{
  size_type count = (size_type)allocatedGPUs_.size();
  bool ok = true;
  for (size_type i = 0; i < count; ++i)
  {
    afc::GPUDevice *dev = allocatedGPUs_[i]->Device;
    ok &= resourceManager_.ReleaseGPU(*dev);
  }
  allocatedGPUs_.clear();
  curRoundAllocatedDevs_.clear();
  if (!ok)
  {
    errorSink.Notify(asmm::WarningMessage(
      AXIS_WARN_ID_SCHEDULING_DEVICE_NOT_RELEASED,
      AXIS_WARN_MSG_SCHEDULING_DEVICE_NOT_RELEASED));
  }
}

assg::KernelConfiguration * assg::KernelScheduler::CreateKernelConfig( 
  afc::GPUDevice& gpu, afmo::MemoryMirror& mirroredMem, 
  asmc::MemoryCommand& memCmd, size_type elementsToAllocate )
{
  return new KernelConfiguration(gpu, mirroredMem, memCmd, elementsToAllocate);
}

assg::GPUTask * assg::KernelScheduler::ScheduleGlobal( size_type threadCount, 
  const asmc::MemoryCommand& memInitCmd, GPUSink& errorSink )
{
  GPUTask *task = new GPUTask();
  size_type remainingElements = threadCount;
  size_type relativeIndex = 0;
  uint64 memoryToAllocate = globalMemAllocated_? 0 : 
    System::ModelMemory().GetTotalSize();
  while (remainingElements > 0)
  {
    GPUMetadata *metadata = Allocate();
    if (metadata == nullptr)
    { // not enough resources
      delete task;
      errorSink.Notify(asmm::ErrorMessage(
        AXIS_ERROR_ID_SCHEDULING_UNAVAILABLE_RESOURCE,
        AXIS_ERROR_MSG_SCHEDULING_UNAVAILABLE_RESOURCE));
      return nullptr;
    }
    afc::GPUDevice *gpu = metadata->Device;
    metadata->Used = true;
    uint64 gpuAvailableSpace = 
      gpu->GetTotalMemorySize() - metadata->UsedMemorySpace;    
    size_type elementsToAllocate = remainingElements;
    if (gpuAvailableSpace < memoryToAllocate)
    { // not enough resources
      delete task;
      errorSink.Notify(asmm::ErrorMessage(
        AXIS_ERROR_ID_SCHEDULING_UNAVAILABLE_RESOURCE,
        AXIS_ERROR_MSG_SCHEDULING_UNAVAILABLE_RESOURCE));
      return nullptr;
    }
    if (elementsToAllocate > gpu->GetMaxThreadCount())
    {
      elementsToAllocate = gpu->GetMaxThreadCount();
    }
    metadata->UsedMemorySpace += memoryToAllocate;
    memoryToAllocate = 0; // in order to avoid further thread allocations to fail

    // create kernel config
    afmo::MemoryMirror *mirroredMem;
    if (globalMemAllocated_)
    {
      mirroredMem = new afmo::NullMemory();
    }
    else
    {
      mirroredMem = new afmg::GPUHeapMirror(System::ModelMemory(), *gpu);
    }
    asmc::MemoryCommand& cmd = memInitCmd.Clone();
    KernelConfiguration *config = CreateKernelConfig(*gpu, *mirroredMem, cmd, 
      elementsToAllocate);

    if (config == nullptr)
    { // kernel configuration failed
      delete task;
      errorSink.Notify(asmm::ErrorMessage(AXIS_ERROR_ID_SCHEDULING_FATAL_ERROR,
        AXIS_ERROR_MSG_SCHEDULING_FATAL_ERROR));
      return nullptr;
    }
    task->AddKernel(*config);
    remainingElements -=elementsToAllocate;
    relativeIndex += elementsToAllocate;
    globalMemAllocated_ = true;
  } 
  return task;
}

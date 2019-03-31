#pragma once
#include <vector>
#include "foundation/Axis.SystemBase.hpp"
#include "services/memory/commands/ElementInitializerCommand.hpp"
#include "services/scheduling/gpu/GPUSink.hpp"
#include "foundation/computing/ResourceManager.hpp"
#include "foundation/computing/GPUDevice.hpp"
#include "KernelConfiguration.hpp"
#include "services/memory/MemoryLayout.hpp"

namespace axis { namespace services { namespace scheduling { namespace gpu {

class GPUTask;

/**
 * Manages local GPU resources and schedules kernel configuration for requested
 * tasks.
 */
class KernelScheduler
{
public:

  /**
   * Constructor.
   *
   * @param [in,out] resourceManager The resource manager to use.
   */
  KernelScheduler(axis::foundation::computing::ResourceManager& resourceManager);
  ~KernelScheduler(void);

  /**
   * Tells to reset statistics about GPU occupancy and start calculating 
   * hereafter.
   */
  void BeginGPURound(void);

  /**
   * Tels to start calculating statistics about GPU occupancy.
   */
  void EndGPURound(void);

  /**
   * Schedules a task that requires owning an exclusive memory for its use.
   *
   * @param memInitializerCmd  The command that initialises memory block.
   * @param blockLayout        How each block in the memory range is arranged.
   * @param elementCount       Number of elements (blocks).
   * @param [in,out] errorSink The error sink.
   *
   * @return null if it fails, a configured GPU task, otherwise.
   */
  GPUTask *ScheduleLocal(
    const axis::services::memory::commands::ElementInitializerCommand& memInitializerCmd,
    const axis::services::memory::MemoryLayout& blockLayout, 
    uint64 elementCount, GPUSink& errorSink);

  /**
   * Schedules a task that works with the entire model memory arena.
   *
   * @param threadCount        Number of threads this task will use.
   * @param memCmd             The command that initialises memory block.
   * @param [in,out] errorSink The error sink.
   *
   * @return null if it fails, a configured GPU task, otherwise.
   */
  GPUTask *ScheduleGlobal(size_type threadCount, 
    const axis::services::memory::commands::MemoryCommand& memCmd, 
    GPUSink& errorSink);

  /**
   * Releases all GPU devices used up to now.
   *
   * @param [in,out] errorSink The error sink.
   */
  void ReleaseDevices(GPUSink& errorSink);
  
private:
  class GPUMetadata;
  typedef std::vector<GPUMetadata *> gpu_list;

  GPUMetadata *Allocate(void);
  GPUMetadata *AllocateNewGPU(void);
  GPUMetadata *GetUnusedGPU(void);
  GPUMetadata *GetLeastUsedGPU(void);
  KernelConfiguration *CreateKernelConfig(
    axis::foundation::computing::GPUDevice& gpu,
    axis::foundation::mirroring::MemoryMirror& mirroredMem,
    axis::services::memory::commands::MemoryCommand& memCmd,
    size_type elementsToAllocate);

  axis::foundation::computing::ResourceManager& resourceManager_;
  gpu_list allocatedGPUs_;
  gpu_list curRoundAllocatedDevs_;
  bool globalMemAllocated_;
};

} } } } // namespace axis::services::scheduling::gpu

#pragma once
#include "foundation/axis.SystemBase.hpp"
#include <vector>
#include "foundation/computing/KernelCommand.hpp"

namespace axis { namespace services { namespace scheduling { namespace gpu {

class KernelConfiguration;

/**
 * Represents a group of kernel configurations that works on a common task and 
 * set of data.
 * @sa KernelConfiguration
 */
class GPUTask
{
public:
  GPUTask(void);
  ~GPUTask(void);

  /**
   * Adds a new kernel configuration to the task. Object is now possessed by
   * the task.
   *
   * @param [in,out] kernelConfig The kernel configuration.
   */
  void AddKernel(KernelConfiguration& kernelConfig);

  /**
   * Allocates required memory by all kernel configurations.
   */
  void AllocateMemory(void);

  /**
   * Deallocates memory claimed by all kernel configurations.
   */
  void DeallocateMemory(void);

  /**
   * Copies data from all kernel configurations to the target GPU devices.
   */
  void Mirror(void);

  /**
   * Copies back kernel configuration data from GPU devices to host memory,
   */
  void Restore(void);

  /**
   * Asks all associated kernel configuration to initialise its data.
   */
  void InitMemory(void);

  /**
   * Claims a system memory block for joint use with GPU devices associated with 
   * this task.
   *
   * @param [in,out] baseAddress Base address of the memory block.
   * @param blockSize            Size of the block.
  **/
  void ClaimMemory(void *baseAddress, uint64 blockSize);

  /**
   * Sets a system memory block so that GPU devices of this task cannot access 
   * it anymore.
   *
   * @param [in,out] baseAddress Base address of the memory block.
  **/
  void ReturnMemory(void *baseAddress);

  /**
   * Executes a command in all associated kernel configurations.
   *
   * @param [in,out] command The command.
   */
  void RunCommand(axis::foundation::computing::KernelCommand& command);

  /**
   * Blocks calling thread until all kernel configurations have finished prior
   * requested tasks.
   */
  void Synchronize(void);

  /**
   * Gets device base memory address associated with a kernel configuration.
   *
   * @param index Zero-based index of the kernel configuration.
   *
   * @return The device base memory address.
   */
  void * GetDeviceMemoryAddress(int index) const;
private:
  typedef std::vector<KernelConfiguration *> kernel_list;
  kernel_list kernels_;
};

} } } } // namespace axis::services::scheduling::gpu

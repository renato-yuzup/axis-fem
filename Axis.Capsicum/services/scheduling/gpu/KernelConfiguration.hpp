#pragma once
#include "foundation/computing/GPUDevice.hpp"
#include "foundation/mirroring/MemoryMirror.hpp"
#include "services/memory/commands/MemoryCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "foundation/computing/KernelCommand.hpp"

namespace axis { namespace services { namespace scheduling { namespace gpu {

/**
 * Represents an expected launch configuration for a GPU kernel.
 */
class KernelConfiguration
{
public:

  /**
   * Constructor.
   *
   * @param [in,out] gpu        The GPU device where kernels should be run.
   * @param [in,out] memMirror  The memory manager that handles memory related
   *                 operations and provides a memory range to work on.
   * @param [in,out] memCommand The command to be issued to initialise memory.
   * @param elementCount        Number of entities to work on.
   */
  KernelConfiguration(axis::foundation::computing::GPUDevice& gpu, 
    axis::foundation::mirroring::MemoryMirror& memMirror, 
    axis::services::memory::commands::MemoryCommand& memCommand,
    size_type elementCount);
  ~KernelConfiguration(void);

  /**
   * Allocates memory for this configuration.
   */
  void Allocate(void);

  /**
   * Deallocates claimed memory.
   */
  void Deallocate(void);

  /**
   * Copies associated memory data to GPU device.
   */
  void Mirror(void);

  /**
   * Copies back associated memory data from GPU device to host memory.
   */
  void Restore(void);

  /**
   * Initialises associated memory data (in host memory).
   */
  void InitMemory(void);

  /**
   * Claims a system memory block for joint use with the GPU device of this 
   * configuration.
   *
   * @param [in,out] baseAddress Base address of the memory block.
   * @param blockSize            Size of the block.
  **/
  void ClaimMemory(void *baseAddress, uint64 blockSize);

  /**
   * Sets a system memory block so that it can no more be accessed by the 
   * GPU device.
   *
   * @param [in,out] baseAddress Base address of the memory block.
  **/
  void ReturnMemory(void *baseAddress);

  /**
   * Executes a command using this kernel configuration.
   *
   * @param [in,out] command The command.
   * @param baseIndex        Starting index of the entities that this command
   *                         in this configuration will work on.
   */
  void RunCommand(axis::foundation::computing::KernelCommand& command, 
    size_type baseIndex);

  /**
   * Returns the number of entities this configuration works on.
   *
   * @return The entity count.
   */
  size_type GetElementCount(void) const;

  /**
   * Blocks calling thread untill all commands previously executed by this
   * configuration finishes.
   */
  void Synchronize(void);

  /**
   * Returns the device base memory address used by this kernel configuration.
   *
   * @return The device base memory address.
   */
  void *GetDeviceMemoryAddress(void) const;
private:
  axis::foundation::computing::GPUDevice *gpu_;
  axis::foundation::mirroring::MemoryMirror *memMirror_;
  axis::services::memory::commands::MemoryCommand *memCommand_;
  size_type elementCount_;
};

} } } } // namespace axis::services::scheduling::gpu

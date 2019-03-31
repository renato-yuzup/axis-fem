#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace services { namespace memory { namespace commands {
  
/**
 * Encapsulates a command to be executed on a single or several similar memory
 * blocks.
**/
class MemoryCommand
{
public:
  MemoryCommand(void);
  virtual ~MemoryCommand(void);  

  /**
   * Operates on the target memory block.
   *
   * @param [in,out] cpuBaseAddress Address of the memory block in host memory.
   *                 Modifications shall occur in this memory address.
   * @param [in,out] gpuBaseAddress Address of the memory block in external
   *                 processing device. Address if provided just as information,
   *                 as it is not accessible due to execution in host memory
   *                 address space.
   */
  virtual void Execute(void *cpuBaseAddress, void *gpuBaseAddress) = 0;

  /**
   * Makes a deep copy of this instance.
   *
   * @return A copy of this instance.
   */
  virtual MemoryCommand& Clone(void) const = 0;
};

} } } } // namespace axis::services::memory::commands

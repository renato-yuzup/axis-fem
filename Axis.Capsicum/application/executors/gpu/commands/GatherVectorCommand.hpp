#pragma once
#include "foundation/computing/KernelCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands {

/**
 * Gathers individual contributions from the output bucket of every finite 
 * element and correctly sums up in the designated global vector.
 */
class GatherVectorCommand : public axis::foundation::computing::KernelCommand
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] modelPtr  Relative pointer to the active 
   *                 ReducedNumericalModel object from which elements will be
   *                 drawn.
   * @param [in,out] vectorPtr Relative pointer to the global vector to be
   *                 used in the gather operation.
   */
  GatherVectorCommand(axis::foundation::memory::RelativePointer& modelPtr,
    axis::foundation::memory::RelativePointer& vectorPtr);
  ~GatherVectorCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );
private:
  axis::foundation::memory::RelativePointer modelPtr_;
  axis::foundation::memory::RelativePointer vectorPtr_;
};

} } } } } // namespace axis::application::executors::gpu::commands

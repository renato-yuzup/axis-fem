#pragma once
#include "foundation/computing/KernelCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands {

/**
 * Stores, in an element metadata array, information of every element regarding 
 * the memory location of the output bucket in the active numerical model.
 */
class InitBucketCommand : public axis::foundation::computing::KernelCommand
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] modelPtr Relative pointer to the ReducedNumericalModel 
   *                 object where elements will be drawn.
   * @param elementBlockSize  Size of total element block, including 
   *                          formulation, material and metadata information.
   */
  InitBucketCommand(axis::foundation::memory::RelativePointer& modelPtr,
    size_type elementBlockSize);
  ~InitBucketCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );
private:
  axis::foundation::memory::RelativePointer modelPtr_;
  size_type elementBlockSize_;
};

} } } } } // namespace axis::application::executors::gpu::commands

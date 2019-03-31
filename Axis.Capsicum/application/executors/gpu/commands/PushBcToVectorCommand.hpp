#pragma once
#include "foundation/computing/KernelCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands {

/**
 * Applies boundary condition values (previously calculated) to a designated
 * global vector.
 */
class PushBcToVectorCommand : public axis::foundation::computing::KernelCommand
{
public:

  /**
   * Constructor.
   *
   * @param [in,out] vectorPtr Relative pointer to the global vector where
   *                 boundary condition values will be ensured.
   * @param [in,out] vectorMaskPtr Relative pointer to the global vector mask where
   *                 boundary condition states are stored.
   * @param ignoreMask         Tells if the vector mask should be ignored.
   * @param bcBlockSize        Length of data block used by boundary condition
   *                           objects affected by this command. Only custom
   *                           data should be accounted for.
   */
  PushBcToVectorCommand(axis::foundation::memory::RelativePointer& vectorPtr,
    axis::foundation::memory::RelativePointer& vectorMaskPtr, bool ignoreMask,
    int bcBlockSize);
  ~PushBcToVectorCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );
private:
  axis::foundation::memory::RelativePointer vectorPtr_;
  axis::foundation::memory::RelativePointer vectorMaskPtr_;
  bool ignoreMask_;
  int bcBlockSize_;
};

} } } } } // namespace axis::application::executors::gpu::commands

#pragma once
#include "foundation/computing/KernelCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace algorithms {

class UpdateReactionForceCommand : public axis::foundation::computing::KernelCommand
{
public:
  UpdateReactionForceCommand(void);
  virtual ~UpdateReactionForceCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
    const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, void * streamPtr );
  void SetDynamicVectors(const axis::foundation::memory::RelativePointer& reactionForcePtr,
    const axis::foundation::memory::RelativePointer& externalLoadPtr,
    const axis::foundation::memory::RelativePointer& internalForcePtr);
private:
  axis::foundation::memory::RelativePointer reaction_;
  axis::foundation::memory::RelativePointer external_;
  axis::foundation::memory::RelativePointer internal_;
};

} } } // namespace axis::domain::algorithms

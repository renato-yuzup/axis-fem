#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace domain { namespace algorithms {

extern void UpdateReactionForceOnGPU(axis::foundation::memory::RelativePointer& reactionForcePtr, 
  axis::foundation::memory::RelativePointer& externalLoadPtr, 
  axis::foundation::memory::RelativePointer& internalLoadPtr, uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr );

} } } // namespace axis::domain::algorithms

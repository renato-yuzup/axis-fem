#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace domain { namespace boundary_conditions {

void ApplyLockOnGPU(uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, real time,
  axis::foundation::memory::RelativePointer vectorMaskPtr);

} } } // namespace axis::domain::boundary_conditions

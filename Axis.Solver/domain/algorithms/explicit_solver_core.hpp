#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace domain { namespace algorithms {

extern void RunExplicitBeforeStepOnGPU(uint64 numThreads, uint64 startIndex, 
  axis::foundation::memory::RelativePointer modelPtr,
  axis::foundation::memory::RelativePointer gpuMaskPtr, 
  real t, real dt, long iterationIndex, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr);

extern void RunExplicitAfterStepOnGPU(uint64 numThreads, uint64 startIndex, 
  axis::foundation::memory::RelativePointer modelPtr,
  axis::foundation::memory::RelativePointer lumpedMassPtr, 
  axis::foundation::memory::RelativePointer gpuMaskPtr, 
  real t, real dt, long iterationIndex, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr);

} } } // namespace axis::domain::algorithms

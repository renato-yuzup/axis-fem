#pragma once
#include "Dimension3D.hpp"

namespace axis { namespace domain { namespace curves {

void UpdateCurveOnGPU(uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, real time );

} } } // namespace axis::domain::curves

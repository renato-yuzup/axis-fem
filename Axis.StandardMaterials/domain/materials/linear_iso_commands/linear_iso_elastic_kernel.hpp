#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace domain { namespace materials { namespace linear_iso_commands {

extern void RunLinearIsoElasticOnGPU( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, 
  uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr, 
  real currentTime, real lastTimeIncrement, real nextTimeIncrement );

} } } } // namespace axis::domain::materials::linear_iso_commands

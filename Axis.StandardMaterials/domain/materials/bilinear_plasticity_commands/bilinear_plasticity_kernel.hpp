#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace domain { namespace materials { 
namespace bilinear_plasticity_commands {

extern void RunBiLinearPlasticityOnGPU( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr, 
  real nextTimeIncrement );

} } } } // namespace axis::domain::materials::bilinear_plasticity_commands

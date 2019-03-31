#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace domain { namespace materials { 
namespace neohookean_commands {

extern void RunNeoHookeanOnGPU( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, 
  uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr );

} } } } // namespace axis::domain::materials::neohookean_commands

#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands { namespace kernels {

void RunPushBcVectorOnGPU(uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, 
  axis::foundation::memory::RelativePointer& vectorPtr, 
  axis::foundation::memory::RelativePointer& vectorMaskPtr, 
  bool ignoreMask, int bcBlockSize);

  } } } } } } // namespace axis::application::scheduling::commands::kernels

#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands { namespace kernels {

void RunInitElementBucketOnGPU(uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr,
  axis::foundation::memory::RelativePointer& modelPtr, 
  size_type elementDataBlockSize);

  } } } } } } // namespace axis::application::executors::gpu::commands::kernels

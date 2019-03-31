#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "Dimension3D.hpp"

namespace axis { namespace foundation { namespace computing {

class AXISSYSTEMBASE_API KernelCommand
{
public:
  KernelCommand(void);
  virtual ~KernelCommand(void);

  virtual void Run(uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
                   const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
                   void * streamPtr) = 0;
};

} } } // namespace axis::foundation::computing

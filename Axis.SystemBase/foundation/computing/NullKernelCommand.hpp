#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "KernelCommand.hpp"

namespace axis { namespace foundation { namespace computing {

class AXISSYSTEMBASE_API NullKernelCommand : public KernelCommand
{
public:
  NullKernelCommand(void);
  ~NullKernelCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );
};

} } } // namespace axis::foundation::computing

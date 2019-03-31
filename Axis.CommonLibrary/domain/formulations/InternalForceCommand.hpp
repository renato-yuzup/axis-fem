#pragma once
#include "Foundation/Axis.CommonLibrary.hpp"
#include "foundation/computing/KernelCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace formulations { 

class AXISCOMMONLIBRARY_API InternalForceCommand : 
  public axis::foundation::computing::KernelCommand
{
public:
  InternalForceCommand(void);
  virtual ~InternalForceCommand(void);
  void SetParameters(uint64 elementBlockSize, 
    axis::foundation::memory::RelativePointer& reducedModelPtr,
    real currentTime, real lastTimeIncrement, real nextTimeIncrement);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );
private:
  virtual void DoRun(uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize,
    axis::foundation::memory::RelativePointer& reducedModelPtr,
    real currentTime, real lastTimeIncrement, real nextTimeIncrement) = 0;
  uint64 elementBlockSize_;
  axis::foundation::memory::RelativePointer reducedModelPtr_;
  real currentTime_, lastTimeIncrement_, nextTimeIncrement_;
};

} } } // namespace axis::domain::formulations

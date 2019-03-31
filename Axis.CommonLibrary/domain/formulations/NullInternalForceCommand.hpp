#pragma once
#include "InternalForceCommand.hpp"

namespace axis { namespace domain { namespace formulations {

class NullInternalForceCommand : public InternalForceCommand
{
public:
  NullInternalForceCommand(void);
  ~NullInternalForceCommand(void);
private:
  virtual void DoRun( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize,
    axis::foundation::memory::RelativePointer& reducedModelPtr, 
    real currentTime, real lastTimeIncrement, real nextTimeIncrement );
};

} } } // namespace axis::domain::formulations

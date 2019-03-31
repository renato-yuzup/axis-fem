#pragma once
#include "UpdateStressCommand.hpp"

namespace axis { namespace domain { namespace materials {

class NullUpdateStressCommand : public UpdateStressCommand
{
public:
  NullUpdateStressCommand(void);
  ~NullUpdateStressCommand(void);
private:
  virtual void DoRun( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize,
    axis::foundation::memory::RelativePointer& reducedModelPtr, 
    real currentTime, real lastTimeIncrement, real nextTimeIncrement );
};

} } } // namespace axis::domain::materials

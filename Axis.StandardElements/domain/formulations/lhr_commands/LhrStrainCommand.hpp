#pragma once
#include "domain/formulations/UpdateStrainCommand.hpp"

namespace axis { namespace domain { namespace formulations { namespace lhr_commands {

class LhrStrainCommand : public axis::domain::formulations::UpdateStrainCommand
{
public:
  LhrStrainCommand(void);
  ~LhrStrainCommand(void);
private:
  virtual void DoRun( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize,
    axis::foundation::memory::RelativePointer& reducedModelPtr,
    real currentTime, real lastTimeIncrement, real nextTimeIncrement );
};

} } } } // namespace axis::domain::formulations::lhr_commands

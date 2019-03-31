#pragma once
#include "domain/materials/UpdateStressCommand.hpp"

namespace axis { namespace domain { namespace materials { namespace neohookean_commands {

class NeoHookeanStressCommand : 
  public axis::domain::materials::UpdateStressCommand
{
public:
  NeoHookeanStressCommand(void);
  ~NeoHookeanStressCommand(void);
private:
  virtual void DoRun( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize, 
    axis::foundation::memory::RelativePointer& reducedModelPtr, real currentTime, 
    real lastTimeIncrement, real nextTimeIncrement );
};

} } } } // namespace axis::domain::materials::neohookean_commands

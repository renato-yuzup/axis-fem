#pragma once
#include "foundation/computing/KernelCommand.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace algorithms {

class ExplicitSolverAfterCommand : public axis::foundation::computing::KernelCommand
{
public:
  ExplicitSolverAfterCommand(
    const axis::foundation::memory::RelativePointer& reducedModelPtr,
    const axis::foundation::memory::RelativePointer& lumpedMassMatrixPtr, 
    const axis::foundation::memory::RelativePointer& gpuMaskPtr, real t, 
    real dt, long iterationIndex);
  virtual ~ExplicitSolverAfterCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );  
private:
  axis::foundation::memory::RelativePointer reducedModelPtr_;        
  axis::foundation::memory::RelativePointer lumpedMassPtr_;        
  axis::foundation::memory::RelativePointer gpuMaskPtr_;        
  real time_, dt_;
  long iterationIndex_;
};

} } } // namespace axis::domain::algorithms

#include "ExplicitSolverAfterCommand.hpp"
#include "explicit_solver_core.hpp"

namespace adal = axis::domain::algorithms;
namespace afm = axis::foundation::memory;

adal::ExplicitSolverAfterCommand::ExplicitSolverAfterCommand( 
  const afm::RelativePointer& reducedModelPtr, 
  const afm::RelativePointer& lumpedMassMatrixPtr, 
  const afm::RelativePointer& gpuMaskPtr, real t, real dt, long iterationIndex) :
reducedModelPtr_(reducedModelPtr), lumpedMassPtr_(lumpedMassMatrixPtr),
  gpuMaskPtr_(gpuMaskPtr), time_(t), dt_(dt), iterationIndex_(iterationIndex)
{
  // nothing to do here
}

adal::ExplicitSolverAfterCommand::~ExplicitSolverAfterCommand(void)
{
  // nothing to do here
}

void adal::ExplicitSolverAfterCommand::Run( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  RunExplicitAfterStepOnGPU(numThreadsToUse, startIndex, reducedModelPtr_, 
    lumpedMassPtr_, gpuMaskPtr_, time_, dt_, iterationIndex_, gridDim, 
    blockDim, streamPtr);
}

#include "ExplicitSolverBeforeCommand.hpp"
#include "explicit_solver_core.hpp"

namespace adal = axis::domain::algorithms;
namespace afm = axis::foundation::memory;

adal::ExplicitSolverBeforeCommand::ExplicitSolverBeforeCommand( 
  const afm::RelativePointer& reducedModelPtr, 
  const afm::RelativePointer& gpuMaskPtr, real t, real dt, long iterationIndex) :
reducedModelPtr_(reducedModelPtr), gpuMaskPtr_(gpuMaskPtr), time_(t), dt_(dt), 
  iterationIndex_(iterationIndex)
{
  // nothing to do here
}

adal::ExplicitSolverBeforeCommand::~ExplicitSolverBeforeCommand(void)
{
  // nothing to do here
}

void adal::ExplicitSolverBeforeCommand::Run( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  RunExplicitBeforeStepOnGPU(numThreadsToUse, startIndex, reducedModelPtr_, 
    gpuMaskPtr_, time_, dt_, iterationIndex_, gridDim, 
    blockDim, streamPtr);
}

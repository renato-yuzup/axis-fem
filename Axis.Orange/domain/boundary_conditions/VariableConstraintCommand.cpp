#include "VariableConstraintCommand.hpp"
#include "variable_constraint_kernel.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace afm = axis::foundation::memory;

adbc::VariableConstraintCommand::VariableConstraintCommand(void)
{
  // nothing to do here
}

adbc::VariableConstraintCommand::~VariableConstraintCommand(void)
{
  // nothing to do here
}

void adbc::VariableConstraintCommand::Run( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  real time = GetTime();
  afm::RelativePointer vectorMaskPtr = GetVectorMask();
  UpdateConstraintOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, time, vectorMaskPtr);
}

#include "LockConstraintCommand.hpp"
#include "lock_constraint_kernel.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace afm = axis::foundation::memory;

adbc::LockConstraintCommand::LockConstraintCommand(void)
{
  // nothing to do here
}

adbc::LockConstraintCommand::~LockConstraintCommand(void)
{
  // nothing to do here
}

void adbc::LockConstraintCommand::Run( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  afm::RelativePointer vectorMaskPtr = GetVectorMask();
  real time = GetTime();
  ApplyLockOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, time, vectorMaskPtr);
}
#pragma once
#include "domain/boundary_conditions/BoundaryConditionUpdateCommand.hpp"

namespace axis { namespace domain { namespace boundary_conditions {

class LockConstraintCommand : public BoundaryConditionUpdateCommand
{
public:
  LockConstraintCommand(void);
  ~LockConstraintCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );
};

} } } // namespace axis::domain::boundary_conditions

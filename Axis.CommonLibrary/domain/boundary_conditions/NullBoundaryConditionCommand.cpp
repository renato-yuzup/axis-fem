#include "NullBoundaryConditionCommand.hpp"

namespace adbc = axis::domain::boundary_conditions;

adbc::BoundaryConditionUpdateCommand& adbc::NullBoundaryConditionCommand::GetInstance(void)
{
  static adbc::NullBoundaryConditionCommand command;
  return command;
}

adbc::NullBoundaryConditionCommand::NullBoundaryConditionCommand( void )
{
  // nothing to do here
}

adbc::NullBoundaryConditionCommand::~NullBoundaryConditionCommand( void )
{
  // nothing to do here
} 

void adbc::NullBoundaryConditionCommand::Run( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  // nothing to do -- null implementation
}

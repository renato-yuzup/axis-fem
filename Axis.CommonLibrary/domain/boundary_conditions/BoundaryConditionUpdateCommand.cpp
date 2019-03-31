#include "BoundaryConditionUpdateCommand.hpp"


namespace adbc = axis::domain::boundary_conditions;
namespace afm = axis::foundation::memory;

adbc::BoundaryConditionUpdateCommand::BoundaryConditionUpdateCommand( void )
{
  time_ = 0;
}

adbc::BoundaryConditionUpdateCommand::~BoundaryConditionUpdateCommand( void )
{
  // nothing to do here
}

void adbc::BoundaryConditionUpdateCommand::Configure( real time, 
  afm::RelativePointer& vectorMaskPtr )
{
  time_ = time;
  vectorMaskPtr_ = vectorMaskPtr;
}

real adbc::BoundaryConditionUpdateCommand::GetTime( void ) const
{
  return time_;
}

afm::RelativePointer adbc::BoundaryConditionUpdateCommand::GetVectorMask(void) const
{
  return vectorMaskPtr_;
}

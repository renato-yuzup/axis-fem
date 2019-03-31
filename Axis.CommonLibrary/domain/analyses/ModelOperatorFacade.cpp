#include "ModelOperatorFacade.hpp"
#include "foundation/memory/pointer.hpp"

namespace ada = axis::domain::analyses;
namespace afm = axis::foundation::memory;

ada::ModelOperatorFacade::ModelOperatorFacade( void )
{
  // nothing to do here
}

ada::ModelOperatorFacade::~ModelOperatorFacade( void )
{
  // nothing to do here
}

ada::ReducedNumericalModel& ada::ModelOperatorFacade::GetModel( void )
{
  return absref<ada::ReducedNumericalModel>(modelPtr_);
}

const ada::ReducedNumericalModel& ada::ModelOperatorFacade::GetModel( void ) const
{
  return absref<ada::ReducedNumericalModel>(modelPtr_);
}

void ada::ModelOperatorFacade::SetTargetModel( afm::RelativePointer& modelPtr )
{
  modelPtr_ = modelPtr;
}

afm::RelativePointer ada::ModelOperatorFacade::GetModelPointer( void )
{
  return modelPtr_;
}

const afm::RelativePointer ada::ModelOperatorFacade::GetModelPointer( void ) const
{
  return modelPtr_;
}

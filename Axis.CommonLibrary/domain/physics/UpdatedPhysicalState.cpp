#include "UpdatedPhysicalState.hpp"
#include "InfinitesimalState.hpp"

namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;

adp::UpdatedPhysicalState::UpdatedPhysicalState( InfinitesimalState& state ) :
state_(state)
{
  // nothing to do here
}

adp::UpdatedPhysicalState::~UpdatedPhysicalState( void )
{
  // nothing to do here
}

afb::ColumnVector& adp::UpdatedPhysicalState::Stress( void )
{
  return state_.Stress();
}

const afb::ColumnVector& adp::UpdatedPhysicalState::Stress( void ) const
{
  return state_.Stress();
}

afb::ColumnVector& adp::UpdatedPhysicalState::LastStressIncrement( void )
{
  return state_.LastStressIncrement();
}

const afb::ColumnVector& adp::UpdatedPhysicalState::LastStressIncrement( void ) const
{
  return state_.LastStressIncrement();
}

afb::ColumnVector& adp::UpdatedPhysicalState::PlasticStrain( void )
{
  return state_.PlasticStrain();
}

const afb::ColumnVector& 
  adp::UpdatedPhysicalState::PlasticStrain( void ) const
{
  return state_.PlasticStrain();
}

real& adp::UpdatedPhysicalState::EffectivePlasticStrain( void )
{
  return state_.EffectivePlasticStrain();
}

real adp::UpdatedPhysicalState::EffectivePlasticStrain( void ) const
{
  return state_.EffectivePlasticStrain();
}

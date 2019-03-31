#include "UpdatedPhysicalState.hpp"
#include "InfinitesimalState.hpp"

namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;

GPU_ONLY aydp::UpdatedPhysicalState::UpdatedPhysicalState( InfinitesimalState& state ) :
state_(state)
{
  // nothing to do here
}

GPU_ONLY aydp::UpdatedPhysicalState::~UpdatedPhysicalState( void )
{
  // nothing to do here
}

GPU_ONLY ayfb::ColumnVector& aydp::UpdatedPhysicalState::Stress( void )
{
  return state_.Stress();
}

GPU_ONLY const ayfb::ColumnVector& aydp::UpdatedPhysicalState::Stress( void ) const
{
  return state_.Stress();
}

GPU_ONLY ayfb::ColumnVector& aydp::UpdatedPhysicalState::LastStressIncrement( void )
{
  return state_.LastStressIncrement();
}

GPU_ONLY const ayfb::ColumnVector& aydp::UpdatedPhysicalState::LastStressIncrement( void ) const
{
  return state_.LastStressIncrement();
}

GPU_ONLY ayfb::ColumnVector& aydp::UpdatedPhysicalState::PlasticStrain( void )
{
  return state_.PlasticStrain();
}

GPU_ONLY const ayfb::ColumnVector& aydp::UpdatedPhysicalState::PlasticStrain( void ) const
{
  return state_.PlasticStrain();
}

GPU_ONLY real& aydp::UpdatedPhysicalState::EffectivePlasticStrain( void )
{
  return state_.EffectivePlasticStrain();
}

GPU_ONLY real aydp::UpdatedPhysicalState::EffectivePlasticStrain( void ) const
{
  return state_.EffectivePlasticStrain();
}

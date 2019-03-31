#include "ModelDynamics.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ayda = axis::yuzu::domain::analyses;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayda::ModelDynamics::ModelDynamics( void )
{
  // private implementation; nothing to do here
}

GPU_ONLY ayda::ModelDynamics::~ModelDynamics( void )
{
  // nothing to do here
}

GPU_ONLY void ayda::ModelDynamics::ResetAll( void )
{
	ResetExternalLoad();
	ResetInternalForce();
	ResetReactionForce();
}

GPU_ONLY void ayda::ModelDynamics::ResetExternalLoad( void )
{
	ExternalLoads().ClearAll();
}

GPU_ONLY void ayda::ModelDynamics::ResetInternalForce( void )
{
	InternalForces().ClearAll();
}

GPU_ONLY const ayfb::ColumnVector& ayda::ModelDynamics::ExternalLoads( void ) const
{
  return yabsref<ayfb::ColumnVector>(_loads);
}

GPU_ONLY ayfb::ColumnVector& ayda::ModelDynamics::ExternalLoads( void )
{
  return yabsref<ayfb::ColumnVector>(_loads);
}

GPU_ONLY const ayfb::ColumnVector& ayda::ModelDynamics::InternalForces( void ) const
{
  return yabsref<ayfb::ColumnVector>(_internalForces);
}

GPU_ONLY ayfb::ColumnVector& ayda::ModelDynamics::InternalForces( void )
{
  return yabsref<ayfb::ColumnVector>(_internalForces);
}

GPU_ONLY void ayda::ModelDynamics::ResetReactionForce( void )
{
	ReactionForce().ClearAll();
}

GPU_ONLY const ayfb::ColumnVector& ayda::ModelDynamics::ReactionForce( void ) const
{
  return yabsref<ayfb::ColumnVector>(_effectiveLoad);
}

GPU_ONLY ayfb::ColumnVector& ayda::ModelDynamics::ReactionForce( void )
{
  return yabsref<ayfb::ColumnVector>(_effectiveLoad);
}

GPU_ONLY const ayfm::RelativePointer ayda::ModelDynamics::GetExternalLoadsPointer( void ) const
{
  return _loads;
}

GPU_ONLY ayfm::RelativePointer ayda::ModelDynamics::GetExternalLoadsPointer( void )
{
  return _loads;
}

GPU_ONLY const ayfm::RelativePointer ayda::ModelDynamics::GetInternalForcesPointer( void ) const
{
  return _internalForces;
}

GPU_ONLY ayfm::RelativePointer ayda::ModelDynamics::GetInternalForcesPointer( void )
{
  return _internalForces;
}

GPU_ONLY const ayfm::RelativePointer ayda::ModelDynamics::GetReactionForcePointer( void ) const
{
  return _effectiveLoad;
}

GPU_ONLY ayfm::RelativePointer ayda::ModelDynamics::GetReactionForcePointer( void )
{
  return _effectiveLoad;
}

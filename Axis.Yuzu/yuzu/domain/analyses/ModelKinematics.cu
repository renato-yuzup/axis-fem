#include "ModelKinematics.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ayda = axis::yuzu::domain::analyses;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayda::ModelKinematics::ModelKinematics( void )
{
  // private implementation; nothing to do here
}

GPU_ONLY ayda::ModelKinematics::~ModelKinematics( void )
{
  // nothing to do here
}

GPU_ONLY void ayda::ModelKinematics::ResetAll( void )
{
	ResetAcceleration();
	ResetDisplacement();
  ResetDisplacementIncrement();
	ResetVelocity();
}

GPU_ONLY void ayda::ModelKinematics::ResetAcceleration( void )
{
	Acceleration().ClearAll();
}

GPU_ONLY void ayda::ModelKinematics::ResetVelocity( void )
{
  Velocity().ClearAll();
}

GPU_ONLY void ayda::ModelKinematics::ResetDisplacement( void )
{
  Displacement().ClearAll();
}

GPU_ONLY const ayfb::ColumnVector& ayda::ModelKinematics::Acceleration( void ) const
{
  return yabsref<ayfb::ColumnVector>(acceleration_);
}

GPU_ONLY ayfb::ColumnVector& ayda::ModelKinematics::Acceleration( void )
{
  return yabsref<ayfb::ColumnVector>(acceleration_);
}

GPU_ONLY const ayfb::ColumnVector& ayda::ModelKinematics::Velocity( void ) const
{
  return yabsref<ayfb::ColumnVector>(velocity_);
}

GPU_ONLY ayfb::ColumnVector& ayda::ModelKinematics::Velocity( void )
{
  return yabsref<ayfb::ColumnVector>(velocity_);
}

GPU_ONLY const ayfb::ColumnVector& ayda::ModelKinematics::Displacement( void ) const
{
  return yabsref<ayfb::ColumnVector>(displacement_);
}

GPU_ONLY ayfb::ColumnVector& ayda::ModelKinematics::Displacement( void )
{
  return yabsref<ayfb::ColumnVector>(displacement_);
}

GPU_ONLY const ayfm::RelativePointer ayda::ModelKinematics::GetAccelerationPointer( void ) const
{
  return acceleration_;
}

GPU_ONLY ayfm::RelativePointer ayda::ModelKinematics::GetAccelerationPointer( void )
{
  return acceleration_;
}

GPU_ONLY const ayfm::RelativePointer ayda::ModelKinematics::GetVelocityPointer( void ) const
{
  return velocity_;
}

GPU_ONLY ayfm::RelativePointer ayda::ModelKinematics::GetVelocityPointer( void )
{
  return velocity_;
}

GPU_ONLY const ayfm::RelativePointer ayda::ModelKinematics::GetDisplacementPointer( void ) const
{
  return displacement_;
}

GPU_ONLY ayfm::RelativePointer ayda::ModelKinematics::GetDisplacementPointer( void )
{
  return displacement_;
}

GPU_ONLY const ayfm::RelativePointer ayda::ModelKinematics::GetDisplacementIncrementPointer( void ) const
{
  return displacementIncrement_;
}

GPU_ONLY ayfm::RelativePointer ayda::ModelKinematics::GetDisplacementIncrementPointer( void )
{
  return displacementIncrement_;
}

GPU_ONLY const ayfb::ColumnVector& ayda::ModelKinematics::DisplacementIncrement( void ) const
{
  return yabsref<ayfb::ColumnVector>(displacementIncrement_);
}

GPU_ONLY ayfb::ColumnVector& ayda::ModelKinematics::DisplacementIncrement( void )
{
  return yabsref<ayfb::ColumnVector>(displacementIncrement_);
}

GPU_ONLY void ayda::ModelKinematics::ResetDisplacementIncrement( void )
{
  DisplacementIncrement().ClearAll();
}

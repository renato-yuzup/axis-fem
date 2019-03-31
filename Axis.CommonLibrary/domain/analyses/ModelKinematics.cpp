#include "ModelKinematics.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "System.hpp"
#include "foundation/memory/pointer.hpp"

namespace ada = axis::domain::analyses;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

ada::ModelKinematics::ModelKinematics( void )
{
	displacement_ = NULLPTR;
  displacementIncrement_ = NULLPTR;
	acceleration_ = NULLPTR;
	velocity_ = NULLPTR;
}

ada::ModelKinematics::~ModelKinematics( void )
{
	if (displacement_ != NULLPTR)
  {
    Displacement().Destroy();
    System::ModelMemory().Deallocate(displacement_);
  }
  if (displacementIncrement_ != NULLPTR)
  {
    DisplacementIncrement().Destroy();
    System::ModelMemory().Deallocate(displacementIncrement_);
  }
	if (velocity_ != NULLPTR)
  {
    Velocity().Destroy();
    System::ModelMemory().Deallocate(velocity_);
  }
	if (acceleration_ != NULLPTR)
  {
    Acceleration().Destroy();
    System::ModelMemory().Deallocate(acceleration_);
  }
}

void ada::ModelKinematics::ResetAll( size_type numDofs )
{
	ResetAcceleration(numDofs);
	ResetDisplacement(numDofs);
  ResetDisplacementIncrement(numDofs);
	ResetVelocity(numDofs);
}

void ada::ModelKinematics::ResetAcceleration( size_type numDofs )
{
	if (acceleration_ != NULLPTR)
	{
		Acceleration().Destroy();
	}
	else if (acceleration_ == NULLPTR)
	{
    acceleration_ = afb::ColumnVector::Create(numDofs);
	}
	Acceleration().ClearAll();
}

void ada::ModelKinematics::ResetVelocity( size_type numDofs )
{
  if (velocity_ != NULLPTR)
  {
    Velocity().Destroy();
  }
  else if (velocity_ == NULLPTR)
  {
    velocity_ = afb::ColumnVector::Create(numDofs);
  }
  Velocity().ClearAll();
}

void ada::ModelKinematics::ResetDisplacement( size_type numDofs )
{
  if (displacement_ != NULLPTR)
  {
    Displacement().Destroy();
  }
  else if (displacement_ == NULLPTR)
  {
    displacement_ = afb::ColumnVector::Create(numDofs);
  }
  Displacement().ClearAll();
}

const afb::ColumnVector& ada::ModelKinematics::Acceleration( void ) const
{
  return absref<afb::ColumnVector>(acceleration_);
}

afb::ColumnVector& ada::ModelKinematics::Acceleration( void )
{
  return absref<afb::ColumnVector>(acceleration_);
}

const afb::ColumnVector& ada::ModelKinematics::Velocity( void ) const
{
  return absref<afb::ColumnVector>(velocity_);
}

afb::ColumnVector& ada::ModelKinematics::Velocity( void )
{
  return absref<afb::ColumnVector>(velocity_);
}

const afb::ColumnVector& ada::ModelKinematics::Displacement( void ) const
{
  return absref<afb::ColumnVector>(displacement_);
}

afb::ColumnVector& ada::ModelKinematics::Displacement( void )
{
  return absref<afb::ColumnVector>(displacement_);
}

const afm::RelativePointer ada::ModelKinematics::GetAccelerationPointer( void ) const
{
  return acceleration_;
}

afm::RelativePointer ada::ModelKinematics::GetAccelerationPointer( void )
{
  return acceleration_;
}

const afm::RelativePointer ada::ModelKinematics::GetVelocityPointer( void ) const
{
  return velocity_;
}

afm::RelativePointer ada::ModelKinematics::GetVelocityPointer( void )
{
  return velocity_;
}

const afm::RelativePointer ada::ModelKinematics::GetDisplacementPointer( void ) const
{
  return displacement_;
}

afm::RelativePointer ada::ModelKinematics::GetDisplacementPointer( void )
{
  return displacement_;
}

bool ada::ModelKinematics::IsDisplacementFieldAvailable( void ) const
{
	return displacement_ != NULLPTR;
}

bool ada::ModelKinematics::IsVelocityFieldAvailable( void ) const
{
	return velocity_ != NULLPTR;
}

bool ada::ModelKinematics::IsAccelerationFieldAvailable( void ) const
{
	return acceleration_ != NULLPTR;
}

afm::RelativePointer ada::ModelKinematics::Create( void )
{
  auto ptr = System::ModelMemory().Allocate(sizeof(ModelKinematics));
  new (*ptr) ModelKinematics();
  return ptr;
}

void * ada::ModelKinematics::operator new( size_t, void *ptr )
{
  return ptr;
}

void ada::ModelKinematics::operator delete( void *, void * )
{
  // nothing to do here
}

const afm::RelativePointer ada::ModelKinematics::GetDisplacementIncrementPointer( void ) const
{
  return displacementIncrement_;
}

afm::RelativePointer ada::ModelKinematics::GetDisplacementIncrementPointer( void )
{
  return displacementIncrement_;
}

const afb::ColumnVector& ada::ModelKinematics::DisplacementIncrement( void ) const
{
  return absref<afb::ColumnVector>(displacementIncrement_);
}

afb::ColumnVector& ada::ModelKinematics::DisplacementIncrement( void )
{
  return absref<afb::ColumnVector>(displacementIncrement_);
}

void ada::ModelKinematics::ResetDisplacementIncrement( size_type numDofs )
{
  if (displacementIncrement_ != NULLPTR)
  {
    DisplacementIncrement().Destroy();
  }
  else if (displacementIncrement_ == NULLPTR)
  {
    displacementIncrement_ = afb::ColumnVector::Create(numDofs);
  }
  DisplacementIncrement().ClearAll();
}

bool ada::ModelKinematics::IsDisplacementIncrementFieldAvailable( void ) const
{
  return displacementIncrement_ != NULLPTR;
}


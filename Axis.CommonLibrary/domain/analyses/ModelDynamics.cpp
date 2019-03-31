#include "ModelDynamics.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/memory/pointer.hpp"
#include "System.hpp"

namespace ada = axis::domain::analyses;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

ada::ModelDynamics::ModelDynamics( void )
{
	_loads = NULLPTR;
	_internalForces = NULLPTR;
	_effectiveLoad = NULLPTR;
}

ada::ModelDynamics::~ModelDynamics( void )
{
	if (_loads != NULLPTR)
  {
    ExternalLoads().Destroy();
    System::ModelMemory().Deallocate(_loads);
  }
    
	if (_internalForces != NULLPTR) 
  {
    InternalForces().Destroy();
    System::ModelMemory().Deallocate(_internalForces);
  }
	if (_effectiveLoad != NULLPTR)
  {
    ReactionForce().Destroy();
    System::ModelMemory().Deallocate(_effectiveLoad);
  }
	_loads = NULLPTR;
	_internalForces = NULLPTR;
	_effectiveLoad = NULLPTR;
}

void ada::ModelDynamics::ResetAll( size_type numDofs )
{
	ResetExternalLoad(numDofs);
	ResetInternalForce(numDofs);
	ResetReactionForce(numDofs);
}

void ada::ModelDynamics::ResetExternalLoad( size_type numDofs )
{
	if (_loads == NULLPTR)
	{
    _loads = afb::ColumnVector::Create(numDofs);
	}
	else if (ExternalLoads().Length() != numDofs)
	{
		ExternalLoads().Destroy();
    System::ModelMemory().Deallocate(_loads);
    _loads = afb::ColumnVector::Create(numDofs);
	}
	ExternalLoads().ClearAll();
}

void ada::ModelDynamics::ResetInternalForce( size_type numDofs )
{
	if (_internalForces == NULLPTR)
	{
    _internalForces = afb::ColumnVector::Create(numDofs);
	}
	else if (InternalForces().Length() != numDofs)
	{
		InternalForces().Destroy();
    System::ModelMemory().Deallocate(_internalForces);
    _internalForces = afb::ColumnVector::Create(numDofs);
	}
	InternalForces().ClearAll();
}

const afb::ColumnVector& ada::ModelDynamics::ExternalLoads( void ) const
{
  return absref<afb::ColumnVector>(_loads);
}

afb::ColumnVector& ada::ModelDynamics::ExternalLoads( void )
{
  return absref<afb::ColumnVector>(_loads);
}

bool ada::ModelDynamics::IsExternalLoadAvailable( void ) const
{
	return _loads != NULLPTR;
}

bool ada::ModelDynamics::IsInternalForceAvailable( void ) const
{
	return _internalForces != NULLPTR;
}

const afb::ColumnVector& ada::ModelDynamics::InternalForces( void ) const
{
  return absref<afb::ColumnVector>(_internalForces);
}

afb::ColumnVector& ada::ModelDynamics::InternalForces( void )
{
  return absref<afb::ColumnVector>(_internalForces);
}

void ada::ModelDynamics::ResetReactionForce( size_type numDofs )
{
	if (_effectiveLoad == NULLPTR)
	{
    _effectiveLoad = afb::ColumnVector::Create(numDofs);
	}
	else if (ReactionForce().Length() != numDofs)
	{
		ReactionForce().Destroy();
    System::ModelMemory().Deallocate(_effectiveLoad);
    _effectiveLoad = afb::ColumnVector::Create(numDofs);
	}
	ReactionForce().ClearAll();
}

const afb::ColumnVector& ada::ModelDynamics::ReactionForce( void ) const
{
  return absref<afb::ColumnVector>(_effectiveLoad);
}

afb::ColumnVector& ada::ModelDynamics::ReactionForce( void )
{
  return absref<afb::ColumnVector>(_effectiveLoad);
}

bool ada::ModelDynamics::IsReactionForceAvailable( void ) const
{
	return _effectiveLoad != NULLPTR;
}

const afm::RelativePointer ada::ModelDynamics::GetExternalLoadsPointer( void ) const
{
  return _loads;
}

afm::RelativePointer ada::ModelDynamics::GetExternalLoadsPointer( void )
{
  return _loads;
}

const afm::RelativePointer ada::ModelDynamics::GetInternalForcesPointer( void ) const
{
  return _internalForces;
}

afm::RelativePointer ada::ModelDynamics::GetInternalForcesPointer( void )
{
  return _internalForces;
}

const afm::RelativePointer ada::ModelDynamics::GetReactionForcePointer( void ) const
{
  return _effectiveLoad;
}

afm::RelativePointer ada::ModelDynamics::GetReactionForcePointer( void )
{
  return _effectiveLoad;
}

afm::RelativePointer ada::ModelDynamics::Create( void )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(ModelDynamics));
  new (*ptr) ModelDynamics();
  return ptr;
}

void * ada::ModelDynamics::operator new( size_t, void *ptr )
{
  return ptr;
}

void ada::ModelDynamics::operator delete( void *, void * )
{
  // nothing to do here
}

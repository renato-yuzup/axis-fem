#include "BoundaryCondition.hpp"
#include "NullBoundaryConditionCommand.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;

adbc::BoundaryCondition::BoundaryCondition( ConstraintType type )
{
	_dof = NULL;
	_type = type;
}

adbc::BoundaryCondition::~BoundaryCondition( void )
{
	// nothing to do
}

ade::DoF *adbc::BoundaryCondition::GetDoF(void) const
{
	return _dof;
}

void adbc::BoundaryCondition::SetDoF(ade::DoF *dof)
{
	if (_dof != NULL)
	{
		if (&(_dof->GetBoundaryCondition()) == this && _dof != dof)
		{
			ade::DoF *aux = _dof;
			_dof = NULL;
			aux->RemoveBoundaryCondition();
		}
	}
	_dof = dof;
	if (dof != NULL)
	{
		if(&(dof->GetBoundaryCondition()) != this)
		{
			dof->SetBoundaryCondition(*this);
		}
	}
}

bool adbc::BoundaryCondition::IsLoad( void ) const
{
	return _type == NodalLoad;
}

bool adbc::BoundaryCondition::IsLock( void ) const
{
	return _type == Lock;
}

bool adbc::BoundaryCondition::IsPrescribedDisplacement( void ) const
{
	return _type == PrescribedDisplacement;
}

bool adbc::BoundaryCondition::IsPrescribedVelocity( void ) const
{
	return _type == PrescribedVelocity;
}

real adbc::BoundaryCondition::operator()( real time ) const
{
  return GetValue(time);
}

adbc::BoundaryCondition::ConstraintType adbc::BoundaryCondition::GetType( void ) const
{
	return _type;
}

bool adbc::BoundaryCondition::IsGPUCapable( void ) const
{
  return false;
}

adbc::BoundaryConditionUpdateCommand& adbc::BoundaryCondition::GetUpdateCommand( void )
{
  return NullBoundaryConditionCommand::GetInstance();
}

int adbc::BoundaryCondition::GetGPUDataSize( void ) const
{
  return 0;
}

void adbc::BoundaryCondition::InitGPUData( void *, real& )
{
  // nothing to do here
}

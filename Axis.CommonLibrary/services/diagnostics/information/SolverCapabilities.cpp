#include "SolverCapabilities.hpp"

axis::services::diagnostics::information::SolverCapabilities::SolverCapabilities( const axis::String& solverName, const axis::String& description, bool timeDepedentCap, bool materialCap, bool geometricCap, bool bcCap )
{
	_name = solverName;
	_description = description;
	_timeCap = timeDepedentCap;
	_materialCap = materialCap;
	_geometricCap = geometricCap;
	_bcCap = bcCap;
}

axis::services::diagnostics::information::SolverCapabilities::SolverCapabilities( const SolverCapabilities& other )
{
	Copy(other);
}

void axis::services::diagnostics::information::SolverCapabilities::Copy( const SolverCapabilities& other )
{
	_timeCap = other.DoesSolveTimeDependentProblems();
	_geometricCap = other.DoesAccountGeometricNonlinearity();
	_materialCap = other.DoesAccountMaterialNonlinearity();
	_bcCap = other.DoesAccountBoundaryConditionsNonlinearity();
	_description = other.GetDescription();
	_name = other.GetSolverName();
}

axis::services::diagnostics::information::SolverCapabilities::~SolverCapabilities( void )
{
	// nothing to do here
}

bool axis::services::diagnostics::information::SolverCapabilities::DoesSolveTimeDependentProblems( void ) const
{
	return _timeCap;
}

bool axis::services::diagnostics::information::SolverCapabilities::DoesAccountMaterialNonlinearity( void ) const
{
	return _materialCap;
}

bool axis::services::diagnostics::information::SolverCapabilities::DoesAccountGeometricNonlinearity( void ) const
{
	return _geometricCap;
}

bool axis::services::diagnostics::information::SolverCapabilities::DoesAccountBoundaryConditionsNonlinearity( void ) const
{
	return _bcCap;
}

axis::services::diagnostics::information::SolverCapabilities& axis::services::diagnostics::information::SolverCapabilities::operator =( const SolverCapabilities& other )
{
	Copy(other);
	return *this;
}

axis::String axis::services::diagnostics::information::SolverCapabilities::GetSolverName( void ) const
{
	return _name;
}

axis::String axis::services::diagnostics::information::SolverCapabilities::GetDescription( void ) const
{
	return _description;
}
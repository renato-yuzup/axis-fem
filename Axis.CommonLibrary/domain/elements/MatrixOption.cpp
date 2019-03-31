#include "MatrixOption.hpp"


bool axis::domain::elements::AllMatricesOption::DoesRequestStiffnessMatrix( void ) const
{
	return true;
}

bool axis::domain::elements::AllMatricesOption::DoesRequestConsistentMassMatrix( void ) const
{
	return true;
}

bool axis::domain::elements::AllMatricesOption::DoesRequestLumpedMassMatrix( void ) const
{
	return true;
}

bool axis::domain::elements::StiffnessMatrixOnlyOption::DoesRequestStiffnessMatrix( void ) const
{
	return true;
}

bool axis::domain::elements::StiffnessMatrixOnlyOption::DoesRequestConsistentMassMatrix( void ) const
{
	return false;
}

bool axis::domain::elements::StiffnessMatrixOnlyOption::DoesRequestLumpedMassMatrix( void ) const
{
	return false;
}

bool axis::domain::elements::ConsistentMassOnlyOption::DoesRequestStiffnessMatrix( void ) const
{
	return false;
}

bool axis::domain::elements::ConsistentMassOnlyOption::DoesRequestConsistentMassMatrix( void ) const
{
	return true;
}

bool axis::domain::elements::ConsistentMassOnlyOption::DoesRequestLumpedMassMatrix( void ) const
{
	return false;
}

bool axis::domain::elements::LumpedMassOnlyOption::DoesRequestStiffnessMatrix( void ) const
{
	return false;
}

bool axis::domain::elements::LumpedMassOnlyOption::DoesRequestConsistentMassMatrix( void ) const
{
	return false;
}

bool axis::domain::elements::LumpedMassOnlyOption::DoesRequestLumpedMassMatrix( void ) const
{
	return true;
}

axis::domain::elements::SomeMatricesOption::SomeMatricesOption( bool stiffness, bool consistentMass, bool lumpedMass )
{
	_stiffness = stiffness; _consistent = consistentMass; _lumped = lumpedMass;
}

bool axis::domain::elements::SomeMatricesOption::DoesRequestStiffnessMatrix( void ) const
{
	return _stiffness;
}

bool axis::domain::elements::SomeMatricesOption::DoesRequestConsistentMassMatrix( void ) const
{
	return _consistent;
}

bool axis::domain::elements::SomeMatricesOption::DoesRequestLumpedMassMatrix( void ) const
{
	return _lumped;
}


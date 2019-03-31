#if defined DEBUG || defined _DEBUG

#include "MockSolverFactoryLocator.hpp"
#include "MockSolver.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/management/ServiceLocator.hpp"
#include "MockClockwork.hpp"

using namespace axis::domain::algorithms;
using namespace axis::unit_tests::core;


bool MockSolverFactoryLocator::CanBuild( const axis::String& solverTypeName, 
  const axis::services::language::syntax::evaluation::ParameterList& params, 
  real stepStartTime, real stepEndTime ) const
{
	return (solverTypeName == _T("MOCK_SOLVER")) && params.IsEmpty();
}

bool MockSolverFactoryLocator::CanBuild( const axis::String& solverTypeName, 
  const axis::services::language::syntax::evaluation::ParameterList& params, real stepStartTime, 
  real stepEndTime, const axis::String& clockworkTypeName, 
  const axis::services::language::syntax::evaluation::ParameterList& clockworkParams ) const
{
	// this method is not important to us in the tests as in real situation it does almost the 
	// same tasks as the other polymorphic method
	throw std::exception("The method or operation is not implemented.");
}

axis::domain::algorithms::Solver& MockSolverFactoryLocator::BuildSolver( 
  const axis::String& solverTypeName, 
  const axis::services::language::syntax::evaluation::ParameterList& params, 
  real stepStartTime, real stepEndTime )
{
	if (!((solverTypeName == _T("MOCK_SOLVER")) && params.IsEmpty()))
	{
		throw axis::foundation::NotSupportedException();
	}
	return *new MockSolver(*new MockClockwork(1));
}

axis::domain::algorithms::Solver& MockSolverFactoryLocator::BuildSolver( 
  const axis::String& solverTypeName, 
  const axis::services::language::syntax::evaluation::ParameterList& params, real stepStartTime, 
  real stepEndTime, axis::domain::algorithms::Clockwork& clockwork )
{
	// this method is not important to us in the tests as in real situation it does almost the 
	// same tasks as the other polymorphic method
	throw std::exception("The method or operation is not implemented.");
}

void MockSolverFactoryLocator::RegisterFactory( 
  axis::application::factories::algorithms::SolverFactory& factory )
{
	throw std::exception("The method or operation is not implemented.");
}

void MockSolverFactoryLocator::UnregisterFactory( 
  axis::application::factories::algorithms::SolverFactory& factory )
{
	throw std::exception("The method or operation is not implemented.");
}

const char * MockSolverFactoryLocator::GetFeaturePath( void ) const
{
	return axis::services::management::ServiceLocator::GetSolverLocatorPath();
}

const char * MockSolverFactoryLocator::GetFeatureName( void ) const
{
	return "MockSolverLocator";
}

#endif

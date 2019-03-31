#if defined DEBUG || defined _DEBUG 

#include "MockSolverFactory.hpp"
#include "MockClockwork.hpp"
#include "MockSolver.hpp"

bool axis::unit_tests::physalis::MockSolverFactory::CanBuild( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime ) const
{
  return analysisType == _T("MOCK_SOLVER");
}

bool axis::unit_tests::physalis::MockSolverFactory::CanBuild( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime, const axis::String& clockworkTypeName, const axis::services::language::syntax::evaluation::ParameterList& clockworkParams ) const
{
  return analysisType == _T("MOCK_SOLVER");
}

axis::domain::algorithms::Solver& axis::unit_tests::physalis::MockSolverFactory::BuildSolver( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime )
{
  return *new MockSolver(*new MockClockwork(1));
}

axis::domain::algorithms::Solver& axis::unit_tests::physalis::MockSolverFactory::BuildSolver( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime, axis::domain::algorithms::Clockwork& clockwork )
{
  return *new MockSolver(*new MockClockwork(1));
}

void axis::unit_tests::physalis::MockSolverFactory::Destroy( void ) const
{
  delete this;
}

#endif
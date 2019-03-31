#include "LinearStaticCGSolverFactory.hpp"
#include "domain/algorithms/CPULinearConjugateGradientSolver.hpp"
#include "domain/algorithms/RegularClockwork.hpp"
#include "RegularClockworkFactory.hpp"

using namespace axis::domain::algorithms;

axis::application::factories::algorithms::LinearStaticCGSolverFactory::LinearStaticCGSolverFactory( void )
{
	// nothing to do here
}

axis::application::factories::algorithms::LinearStaticCGSolverFactory::~LinearStaticCGSolverFactory( void )
{
	// nothing to do here
}

bool axis::application::factories::algorithms::LinearStaticCGSolverFactory::CanBuild( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real, real ) const
{
	/*
		This is our requirements:
			- Solver type: LINEAR_STATIC
			- Start time <= end time (more importantly, not equal)
			- No solver parameters
	*/

	return	analysisType == _T("LINEAR_STATIC") &&
		paramList.IsEmpty();
}

bool axis::application::factories::algorithms::LinearStaticCGSolverFactory::CanBuild( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime, const axis::String& clockworkTypeName, const axis::services::language::syntax::evaluation::ParameterList& clockworkParams ) const
{
	// we can only use the regular clockwork type
	if (clockworkTypeName != RegularClockworkFactory::ClockworkTypeName) return false;

	// we will not check if clockwork parameters are valid; postpone it to the step parser
	return CanBuild(analysisType, paramList, stepStartTime, stepEndTime);
}

axis::domain::algorithms::Solver& axis::application::factories::algorithms::LinearStaticCGSolverFactory::BuildSolver( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime )
{
	Clockwork& cw = RegularClockwork::Create(stepEndTime - stepStartTime);
	return *new axis::domain::algorithms::CPULinearConjugateGradientSolver(cw);
}

axis::domain::algorithms::Solver& axis::application::factories::algorithms::LinearStaticCGSolverFactory::BuildSolver( const axis::String& analysisType, const axis::services::language::syntax::evaluation::ParameterList& paramList, real stepStartTime, real stepEndTime, axis::domain::algorithms::Clockwork& clockwork )
{
	return *new axis::domain::algorithms::CPULinearConjugateGradientSolver(clockwork);
}

void axis::application::factories::algorithms::LinearStaticCGSolverFactory::Destroy( void ) const
{
	delete this;
}


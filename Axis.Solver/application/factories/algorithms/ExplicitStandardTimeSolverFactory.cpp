#include "ExplicitStandardTimeSolverFactory.hpp"
#include "domain/algorithms/WaveSpeedProportionalClockwork.hpp"
#include "domain/algorithms/ExplicitDynamicSolver.hpp"

namespace aafa = axis::application::factories::algorithms;
namespace adal = axis::domain::algorithms;
namespace aslse = axis::services::language::syntax::evaluation;

bool aafa::ExplicitStandardTimeSolverFactory::CanBuild( const axis::String& analysisType, 
                                                        const aslse::ParameterList& paramList, 
                                                        real stepStartTime, real stepEndTime ) const
{
	/*
		This is our requirements:
			- Solver type: EXPLICIT_STANDARD
			- Start time < end time (more importantly, not equal)
			- No solver parameters
	*/
	
	if (analysisType != _T("EXPLICIT_STANDARD")) return false;
	if (abs((stepStartTime - stepEndTime)) <= 1e-14) return false;
	if (!paramList.IsEmpty()) return false;

	return true;
}

bool aafa::ExplicitStandardTimeSolverFactory::CanBuild( const axis::String& analysisType, 
                                                        const aslse::ParameterList& paramList, 
                                                        real stepStartTime, real stepEndTime, 
                                                        const axis::String&, 
                                                        const aslse::ParameterList& ) const
{
	// we accept any clockwork type; it might not work as expected, 
	// though, but this is not our problem... ^^
	return CanBuild(analysisType, paramList, stepStartTime, stepEndTime);
}

adal::Solver& aafa::ExplicitStandardTimeSolverFactory::BuildSolver( const axis::String& analysisType, 
                                                                    const aslse::ParameterList& paramList, 
                                                                    real stepStartTime, real stepEndTime )
{
	adal::Clockwork& cw = *new adal::WaveSpeedProportionalClockwork(0.9, true);
	return *new adal::ExplicitDynamicSolver(cw);
}

adal::Solver& aafa::ExplicitStandardTimeSolverFactory::BuildSolver( const axis::String& analysisType, 
                                                                    const aslse::ParameterList& paramList, 
                                                                    real stepStartTime, real stepEndTime, 
                                                                    adal::Clockwork& clockwork )
{
	return *new adal::ExplicitDynamicSolver(clockwork);
}

void aafa::ExplicitStandardTimeSolverFactory::Destroy( void ) const
{
	delete this;
}

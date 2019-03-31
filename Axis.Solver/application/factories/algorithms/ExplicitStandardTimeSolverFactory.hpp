#pragma once
#include "application/factories/algorithms/SolverFactory.hpp"

namespace axis { namespace application { namespace factories { namespace algorithms {

class ExplicitStandardTimeSolverFactory : public SolverFactory
{
public:
	virtual bool CanBuild( const axis::String& analysisType, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime ) const;
	virtual bool CanBuild( const axis::String& analysisType, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime, const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& clockworkParams ) const;
	virtual axis::domain::algorithms::Solver& BuildSolver( const axis::String& analysisType, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime );
	virtual axis::domain::algorithms::Solver& BuildSolver( const axis::String& analysisType, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime, axis::domain::algorithms::Clockwork& clockwork );
	virtual void Destroy( void ) const;
};

} } } } // namespace axis::application::factories::algorithms

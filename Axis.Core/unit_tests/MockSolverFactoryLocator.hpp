#pragma once
#include "application/locators/SolverFactoryLocator.hpp"

/**************************************************************************************************
 * <summary>	Just a mock object to use in substitute for the standard solver locator 
 * 				in our test cases. </summary>
 **************************************************************************************************/
class MockSolverFactoryLocator : public axis::application::locators::SolverFactoryLocator
{
public:
	virtual bool CanBuild( const axis::String& solverTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& params, 
    real stepStartTime, real stepEndTime ) const;
	virtual bool CanBuild( const axis::String& solverTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& params, 
    real stepStartTime, real stepEndTime, const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& clockworkParams ) const;
	virtual axis::domain::algorithms::Solver& BuildSolver( const axis::String& solverTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& params, 
    real stepStartTime, real stepEndTime );
	virtual axis::domain::algorithms::Solver& BuildSolver( const axis::String& solverTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& params, 
    real stepStartTime, real stepEndTime, axis::domain::algorithms::Clockwork& clockwork );
	virtual void RegisterFactory( axis::application::factories::algorithms::SolverFactory& factory );
	virtual void UnregisterFactory( axis::application::factories::algorithms::SolverFactory& factory );
	virtual const char * GetFeaturePath( void ) const;
	virtual const char * GetFeatureName( void ) const;
};


#pragma once
#include "application/factories/algorithms/ClockworkFactory.hpp"

namespace axis { namespace application { namespace factories { namespace algorithms {

class RegularClockworkFactory : public ClockworkFactory
{
public:
	virtual void Destroy( void ) const;
	virtual bool CanBuild( const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime ) const;
	virtual axis::domain::algorithms::Clockwork& Build( const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime );
	static const axis::String::char_type * ClockworkTypeName;
};

} } } } // namespace axis::application::factories::algorithms

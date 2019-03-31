#pragma once
#include "ParserAction.hpp"

namespace axis { namespace services { namespace language { namespace actions {

class NopAction : public ParserAction
{
public:
	virtual void Run( const axis::services::language::parsing::ParseResult& result ) const;
	virtual ParserAction& Clone( void ) const;
};					

} } } } // namespace axis::services::language::actions

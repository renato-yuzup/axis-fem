#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../parsing/ParseResult.hpp"

namespace axis { namespace services { namespace language { namespace actions {

class AXISMINT_API ParserAction
{
public:
	/* Allow inheritance and correct destructor call */
	virtual ~ParserAction(void);

	/*
		Execute this action passing the value extracted by the lexer.
	*/
	virtual void Run(const axis::services::language::parsing::ParseResult& result) const = 0;

	/*
		Creates a copy of this object.
	*/
	virtual ParserAction& Clone(void) const = 0;
};					

} } } } // namespace axis::services::language::actions

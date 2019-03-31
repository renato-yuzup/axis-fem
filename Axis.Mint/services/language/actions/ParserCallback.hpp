#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../parsing/ParseResult.hpp"

namespace axis { namespace services { namespace language { namespace actions {

class AXISMINT_API ParserCallback
{
public:
	virtual ~ParserCallback(void);
	virtual void ProcessLexerSuccessEvent(
    const axis::services::language::parsing::ParseResult& result);
	virtual void ProcessLexerSuccessEvent(
    const axis::services::language::parsing::ParseResult& result, int tag);
	virtual void ProcessLexerSuccessEvent(
    const axis::services::language::parsing::ParseResult& result, void *data);
};
			
} } } } // namespace axis::services::language::actions

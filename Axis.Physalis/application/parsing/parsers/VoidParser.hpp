#pragma once
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/parsers/BlockParser.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class VoidParser : public axis::application::parsing::parsers::BlockParser
{
private:
	const axis::application::factories::parsers::BlockProvider& _parent;
public:
	VoidParser(const axis::application::factories::parsers::BlockProvider& factory);
	virtual ~VoidParser(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext(
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
};				

} } } } // namespace axis::application::parsing::parsers

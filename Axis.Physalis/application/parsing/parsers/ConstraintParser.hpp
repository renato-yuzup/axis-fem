#pragma once
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/locators/ConstraintParserLocator.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class ConstraintParser : public axis::application::parsing::parsers::BlockParser
{
public:
	ConstraintParser(axis::application::locators::ConstraintParserLocator& locator);
	virtual ~ConstraintParser(void);

	virtual BlockParser& GetNestedContext( const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );

	virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
private:
  axis::application::locators::ConstraintParserLocator& _locator;
};

} } } } // namespace axis::application::parsing::parsers

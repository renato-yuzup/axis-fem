#pragma once

#include "application/parsing/parsers/BlockParser.hpp"
#include "application/locators/ElementParserLocator.hpp"
#include "AxisString.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class ElementParser : public axis::application::parsing::parsers::BlockParser
{
public:
	ElementParser(axis::application::locators::ElementParserLocator& parentProvider, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	~ElementParser(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	virtual void DoCloseContext( void );
	virtual void DoStartContext( void );
private:
  axis::application::parsing::parsers::BlockParser *_innerParser;
  void TriggerError(const axis::String& errorMsg, const int errorId) const;
  axis::domain::collections::ElementSet& EnsureElementSet(const axis::String& elementSetId) const;

  axis::application::locators::ElementParserLocator& _parentProvider;
  axis::String _elementSetId;
};				

} } } } // namespace axis::application::parsing::parsers

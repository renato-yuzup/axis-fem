#pragma once
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/factories/parsers/BlockProvider.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class SolverParser : public axis::application::parsing::parsers::BlockParser
{
public:
	SolverParser(axis::application::factories::parsers::BlockProvider& parentProvider, 
    axis::application::parsing::parsers::BlockParser& innerParser);
	virtual ~SolverParser(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
protected:
	virtual void DoCloseContext( void );
	virtual void DoStartContext( void );
private:
	axis::application::factories::parsers::BlockProvider& _parentProvider;
	axis::application::parsing::parsers::BlockParser& _innerParser;
};

} } } } // namespace axis::application::parsing::parsers

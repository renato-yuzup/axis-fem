#pragma once
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/locators/SolverFactoryLocator.hpp"
#include "application/factories/parsers/AnalysisBlockParserProvider.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class AnalysisBlockParser : public axis::application::parsing::parsers::BlockParser
{
public:
  AnalysisBlockParser(
    axis::application::factories::parsers::AnalysisBlockParserProvider& parentProvider);
  virtual ~AnalysisBlockParser(void);
  virtual BlockParser& GetNestedContext( const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
  virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
  virtual void DoStartContext( void );
  virtual void DoCloseContext( void );
private:
	axis::application::factories::parsers::AnalysisBlockParserProvider& _parentProvider;
};

} } } } // namespace axis::application::parsing::parsers

#pragma once
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/locators/SolverFactoryLocator.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

class AnalysisBlockParserProvider : public axis::application::factories::parsers::BlockProvider
{
private:
	axis::application::locators::SolverFactoryLocator *_solverLocator;
public:
	AnalysisBlockParserProvider(void);
	~AnalysisBlockParserProvider(void);
	virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual const char * GetFeaturePath( void ) const;
	virtual const char * GetFeatureName( void ) const;
};

} } } } // namespace axis::application::factories::parsers

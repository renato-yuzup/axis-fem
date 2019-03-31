#pragma once
#include "application/factories/parsers/BlockProvider.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

class NodalLoadParserProvider : public axis::application::factories::parsers::BlockProvider
{
public:
	NodalLoadParserProvider(void);
	~NodalLoadParserProvider(void);
	virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual const char *GetFeaturePath(void) const;
	virtual const char *GetFeatureName(void) const;
};

} }	} } // naespace axis::application::factories::parsers

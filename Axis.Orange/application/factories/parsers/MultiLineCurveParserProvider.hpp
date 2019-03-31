#pragma once
#include "application/factories/parsers/BlockProvider.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

class MultiLineCurveParserProvider : public axis::application::factories::parsers::BlockProvider
{
public:
	MultiLineCurveParserProvider(void);
	~MultiLineCurveParserProvider(void);
	virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual const char *GetFeaturePath(void) const;
	virtual const char *GetFeatureName(void) const;
private:
  axis::AsciiString _featurePath;
};

} } } } // namespace axis::application::factories::parsers
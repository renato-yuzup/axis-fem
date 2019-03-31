#pragma once
#include "application/factories/parsers/ElementParserFactory.hpp"
#include "application/factories/parsers/BlockProvider.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

class LinearHexahedronSimpleParserFactory : public axis::application::factories::parsers::ElementParserFactory
{
public:
	LinearHexahedronSimpleParserFactory(const axis::application::factories::parsers::BlockProvider& provider);
	~LinearHexahedronSimpleParserFactory(void);
	virtual void Destroy( void ) const;
	virtual bool CanBuild( 
    const axis::application::parsing::core::SectionDefinition& definition ) const;
	virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
      const axis::application::parsing::core::SectionDefinition& definition, 
      axis::domain::collections::ElementSet& elementCollection ) const;
private:
	const axis::application::factories::parsers::BlockProvider& provider_;
};	

} } } } // namespace axis::application::factories::parsers

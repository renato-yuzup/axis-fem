#pragma once
#include "application/factories/parsers/ElementParserFactory.hpp"
#include "application/factories/parsers/BlockProvider.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

  class LinearHexahedronPusoParserFactory : public ElementParserFactory
  {
  public:
    LinearHexahedronPusoParserFactory(const axis::application::factories::parsers::BlockProvider& provider);
    ~LinearHexahedronPusoParserFactory(void);
    virtual void Destroy( void ) const;
    virtual bool CanBuild( const axis::application::parsing::core::SectionDefinition& definition ) const;
    virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
      const axis::application::parsing::core::SectionDefinition& definition, 
      axis::domain::collections::ElementSet& elementCollection ) const;
  private:
    const axis::application::factories::parsers::BlockProvider& provider_;
  };	

} } } } // namespace axis::application::factories::parsers

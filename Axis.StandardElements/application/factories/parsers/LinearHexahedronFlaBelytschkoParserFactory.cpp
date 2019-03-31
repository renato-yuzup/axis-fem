#include "LinearHexahedronFlaBelytschkoParserFactory.hpp"
#include "application/factories/elements/LinearFlanaganBelytschkoHexaFactory.hpp"
#include "application/parsing/parsers/HexahedronElementParser.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;

aafp::LinearHexahedronFlaBelytschkoParserFactory::LinearHexahedronFlaBelytschkoParserFactory( 
  const BlockProvider& provider ) : provider_(provider)
{
  // nothing to do here
}

aafp::LinearHexahedronFlaBelytschkoParserFactory::~LinearHexahedronFlaBelytschkoParserFactory( void )
{
  // nothing to do here
}

void aafp::LinearHexahedronFlaBelytschkoParserFactory::Destroy( void ) const
{
  delete this;
}

bool aafp::LinearHexahedronFlaBelytschkoParserFactory::CanBuild( 
  const aapc::SectionDefinition& definition ) const
{
  return aafe::LinearFlanaganBelytschkoHexaFactory::IsValidDefinition(definition);
}

aapps::BlockParser& aafp::LinearHexahedronFlaBelytschkoParserFactory::BuildParser( 
  const aapc::SectionDefinition& definition, adc::ElementSet& elementCollection ) const
{
  aafe::HexahedronFactory *factory = new aafe::LinearFlanaganBelytschkoHexaFactory();
  return *new aapps::HexahedronElementParser(provider_, definition, elementCollection, *factory);
}

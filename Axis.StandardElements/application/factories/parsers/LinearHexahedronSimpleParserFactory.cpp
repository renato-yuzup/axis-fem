#include "LinearHexahedronSimpleParserFactory.hpp"
#include "application/factories/elements/LinearSimpleHexahedronFactory.hpp"
#include "application/parsing/parsers/HexahedronElementParser.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;

aafp::LinearHexahedronSimpleParserFactory::LinearHexahedronSimpleParserFactory( 
  const BlockProvider& provider ) : provider_(provider)
{
	// nothing to do here
}

aafp::LinearHexahedronSimpleParserFactory::~LinearHexahedronSimpleParserFactory( void )
{
	// nothing to do here
}

void aafp::LinearHexahedronSimpleParserFactory::Destroy( void ) const
{
  delete this;
}

bool aafp::LinearHexahedronSimpleParserFactory::CanBuild( 
  const aapc::SectionDefinition& definition ) const
{
  return aafe::LinearSimpleHexahedronFactory::IsValidDefinition(definition);
}

aapps::BlockParser& aafp::LinearHexahedronSimpleParserFactory::BuildParser( 
  const aapc::SectionDefinition& definition, adc::ElementSet& elementCollection ) const
{
  aafe::HexahedronFactory *factory = new aafe::LinearSimpleHexahedronFactory();
  return *new aapps::HexahedronElementParser(provider_, definition, elementCollection, *factory);
}

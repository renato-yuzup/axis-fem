#include "NonLinearHexahedronSimpleParserFactory.hpp"
#include "application/factories/elements/NonLinearSimpleHexahedronFactory.hpp"
#include "application/parsing/parsers/HexahedronElementParser.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;

aafp::NonLinearHexahedronSimpleParserFactory::NonLinearHexahedronSimpleParserFactory( 
  const BlockProvider& provider ) : provider_(provider)
{
	// nothing to do here
}

aafp::NonLinearHexahedronSimpleParserFactory::~NonLinearHexahedronSimpleParserFactory( void )
{
	// nothing to do here
}

void aafp::NonLinearHexahedronSimpleParserFactory::Destroy( void ) const
{
  delete this;
}

bool aafp::NonLinearHexahedronSimpleParserFactory::CanBuild( 
  const aapc::SectionDefinition& definition ) const
{
  return aafe::NonLinearSimpleHexahedronFactory::IsValidDefinition(definition);
}

aapps::BlockParser& aafp::NonLinearHexahedronSimpleParserFactory::BuildParser( 
  const aapc::SectionDefinition& definition, adc::ElementSet& elementCollection ) const
{
  aafe::HexahedronFactory *factory = new aafe::NonLinearSimpleHexahedronFactory();
  return *new aapps::HexahedronElementParser(provider_, definition, elementCollection, *factory);
}

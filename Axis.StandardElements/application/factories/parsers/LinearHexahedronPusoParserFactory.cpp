#include "LinearHexahedronPusoParserFactory.hpp"
#include "application/factories/elements/LinearPusoHexahedronFactory.hpp"
#include "application/parsing/parsers/HexahedronElementParser.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;

aafp::LinearHexahedronPusoParserFactory::LinearHexahedronPusoParserFactory( 
  const BlockProvider& provider ) : provider_(provider)
{
  // nothing to do here
}

aafp::LinearHexahedronPusoParserFactory::~LinearHexahedronPusoParserFactory( void )
{
  // nothing to do here
}

void aafp::LinearHexahedronPusoParserFactory::Destroy( void ) const
{
  delete this;
}

bool aafp::LinearHexahedronPusoParserFactory::CanBuild( 
  const aapc::SectionDefinition& definition ) const
{
  return aafe::LinearPusoHexahedronFactory::IsValidDefinition(definition);
}

aapps::BlockParser& aafp::LinearHexahedronPusoParserFactory::BuildParser( 
  const aapc::SectionDefinition& definition, adc::ElementSet& elementCollection ) const
{
  aafe::HexahedronFactory *factory = new aafe::LinearPusoHexahedronFactory();
  return *new aapps::HexahedronElementParser(provider_, definition, elementCollection, *factory);
}

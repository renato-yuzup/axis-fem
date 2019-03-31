#include "NonLinearHexahedronFlaBelytschkoParserFactory.hpp"
#include "application/factories/elements/NonLinearFlanaganBelytschkoHexaFactory.hpp"
#include "application/parsing/parsers/HexahedronElementParser.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;

aafp::NonLinearHexahedronFlaBelytschkoParserFactory::NonLinearHexahedronFlaBelytschkoParserFactory( 
	const BlockProvider& provider ) : provider_(provider)
{
	// nothing to do here
}

aafp::NonLinearHexahedronFlaBelytschkoParserFactory::~NonLinearHexahedronFlaBelytschkoParserFactory( void )
{
	// nothing to do here
}

void aafp::NonLinearHexahedronFlaBelytschkoParserFactory::Destroy( void ) const
{
	delete this;
}

bool aafp::NonLinearHexahedronFlaBelytschkoParserFactory::CanBuild( 
	const aapc::SectionDefinition& definition ) const
{
	return aafe::NonLinearFlanaganBelytschkoHexaFactory::IsValidDefinition(definition);
}

aapps::BlockParser& aafp::NonLinearHexahedronFlaBelytschkoParserFactory::BuildParser( 
	const aapc::SectionDefinition& definition, adc::ElementSet& elementCollection ) const
{
	aafe::HexahedronFactory *factory = new aafe::NonLinearFlanaganBelytschkoHexaFactory();
	return *new aapps::HexahedronElementParser(provider_, definition, elementCollection, *factory);
}

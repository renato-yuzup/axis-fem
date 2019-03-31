#include "RootParserProvider.hpp"
#include "application/parsing/parsers/RootParser.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "services/management/ServiceLocator.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;

aafp::RootParserProvider::RootParserProvider(void)
{
	// do nothing
}

aafp::RootParserProvider::~RootParserProvider(void)
{
	// do nothing
}

const char * aafp::RootParserProvider::GetFeaturePath( void ) const
{
	return axis::services::management::ServiceLocator::GetMasterInputParserProviderPath();
}

const char * aafp::RootParserProvider::GetFeatureName( void ) const
{
	return "NumericalAnalysisParserProvider";
}

bool aafp::RootParserProvider::CanParse( const axis::String& blockName, 
                                         const aslse::ParameterList& params )
{
	// we can't parse anything, we are just a container
	return false;
}

aapps::BlockParser& aafp::RootParserProvider::BuildParser( const axis::String& contextName, 
                                                           const aslse::ParameterList& params )
{
	return *new aapps::RootParser(*this);
}

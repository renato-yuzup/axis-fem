#include "NodalLoadParserProvider.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "application/parsing/parsers/NodalLoadParser.hpp"
#include "services/management/ServiceLocator.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;

aafp::NodalLoadParserProvider::NodalLoadParserProvider( void )
{
	// nothing to do here
}

aafp::NodalLoadParserProvider::~NodalLoadParserProvider( void )
{
	// nothing to do here
}

bool aafp::NodalLoadParserProvider::CanParse( const axis::String& blockName, 
                                              const aslse::ParameterList& paramList )
{
	return (blockName.equals(_T("NODAL_LOADS")) && paramList.Count() == 0);
}

aapps::BlockParser& aafp::NodalLoadParserProvider::BuildParser( const axis::String& contextName, 
                                                                const aslse::ParameterList& paramList )
{
	if (!CanParse(contextName, paramList))
	{
		throw axis::foundation::InvalidOperationException();
	}
	return *new aapps::NodalLoadParser(*this);
}

const char * aafp::NodalLoadParserProvider::GetFeaturePath( void ) const
{
	return asmg::ServiceLocator::GetNodalLoadInputParserProviderPath();
}

const char * aafp::NodalLoadParserProvider::GetFeatureName( void ) const
{
	return "StandardNodalLoadParserProvider";
}

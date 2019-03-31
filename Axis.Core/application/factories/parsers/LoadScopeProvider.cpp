#include "LoadScopeProvider.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/parsing/parsers/EmptyBlockParser.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;
namespace afd = axis::foundation::definitions;

bool aafp::LoadScopeProvider::CanParse( const axis::String& blockName, 
                                        const aslse::ParameterList& paramList )
{
	return blockName.compare(afd::AxisInputLanguage::LoadSectionBlockName) == 0 && paramList.Count() == 0;
}

aapps::BlockParser& aafp::LoadScopeProvider::BuildParser( const axis::String& contextName, 
                                                          const aslse::ParameterList& paramList )
{
	if (!CanParse(contextName, paramList))
	{
		throw axis::foundation::InvalidOperationException();
	}
	return *new aapps::EmptyBlockParser(*this);
}

const char * aafp::LoadScopeProvider::GetFeaturePath( void ) const
{
	return asmg::ServiceLocator::GetLoadSectionInputParserProviderPath();
}

const char * aafp::LoadScopeProvider::GetFeatureName( void ) const
{
	return "InputFileLoadSectionParserProvider";
}

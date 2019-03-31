#include "CurveScopeProvider.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/parsing/parsers/EmptyBlockParser.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;
namespace afd = axis::foundation::definitions;

bool aafp::CurveScopeProvider::CanParse( const axis::String& blockName, 
                                         const aslse::ParameterList& paramList )
{
	return blockName.compare(afd::AxisInputLanguage::CurveSectionBlockName) == 0 && 
         paramList.Count() == 0;
}

aapps::BlockParser& aafp::CurveScopeProvider::BuildParser( const axis::String& contextName, 
  const aslse::ParameterList& paramList )
{
	return *new aapps::EmptyBlockParser(*this);
}

const char * aafp::CurveScopeProvider::GetFeaturePath( void ) const
{
	return asmg::ServiceLocator::GetCurveSectionInputParserProviderPath();
}

const char * aafp::CurveScopeProvider::GetFeatureName( void ) const
{
	return "InputFileCurveSectionParserProvider";
}

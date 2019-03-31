#include "AnalysisBlockParserProvider.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "application/parsing/parsers/EmptyBlockParser.hpp"
#include "services/management/ServiceLocator.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "application/parsing/parsers/AnalysisBlockParser.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace afd = axis::foundation::definitions;

aafp::AnalysisBlockParserProvider::AnalysisBlockParserProvider( void )
{
	// nothing to do here
}

aafp::AnalysisBlockParserProvider::~AnalysisBlockParserProvider( void )
{
	// nothing to do here
}

bool aafp::AnalysisBlockParserProvider::CanParse( const axis::String& blockName, 
                                                  const aslse::ParameterList& paramList )
{
	return blockName == afd::AxisInputLanguage::AnalysisSettingsSyntax.BlockName && paramList.IsEmpty();
}

aapps::BlockParser& aafp::AnalysisBlockParserProvider::BuildParser( const axis::String& blockName, 
  const aslse::ParameterList& paramList )
{
	if (!CanParse(blockName, paramList))
	{
		throw axis::foundation::InvalidOperationException();
	}

	// our block is only a container for nested blocks; no
	// instruction is allowed
	return *new aapps::AnalysisBlockParser(*this);
}

const char * aafp::AnalysisBlockParserProvider::GetFeaturePath( void ) const
{
	return "axis.base.input.providers.AnalysisSettingsProvider";
}

const char * aafp::AnalysisBlockParserProvider::GetFeatureName( void ) const
{
	return "AnalysisSettingsProvider";
}

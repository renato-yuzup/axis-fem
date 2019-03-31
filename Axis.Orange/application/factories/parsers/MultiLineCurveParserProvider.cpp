#include "MultiLineCurveParserProvider.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/parsing/parsers/MultiLineCurveParser.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"

namespace aafp = axis::application::factories::parsers;
namespace asmg = axis::services::management;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aapps = axis::application::parsing::parsers;

aafp::MultiLineCurveParserProvider::MultiLineCurveParserProvider( void )
{
	// initialize our feature path
	_featurePath = asmg::ServiceLocator::GetCurveProviderBasePath() + "MultiLineCurveProvider";
}

aafp::MultiLineCurveParserProvider::~MultiLineCurveParserProvider( void )
{
	// nothing to do here
}

bool aafp::MultiLineCurveParserProvider::CanParse( const axis::String& blockName, 
                                                   const aslse::ParameterList& paramList )
{
	// we can only parse multiline curves -- of course, it must have an identifier with it
	if (blockName != _T("MULTILINE_CURVE")) return false;
	if (paramList.Count() != 1 || !paramList.IsDeclared(_T("ID"))) return false;
	if (!paramList.GetParameterValue(_T("ID")).IsAtomic()) return false;
	if (((aslse::AtomicValue&)paramList.GetParameterValue(_T("ID"))).IsNull()) return false;
	// ok, it's our turn
	return true;
}

aapps::BlockParser& aafp::MultiLineCurveParserProvider::BuildParser( 
  const axis::String& contextName, const aslse::ParameterList& paramList )
{
	// if we cannot parse, then we must warn the use
	if (!CanParse(contextName, paramList))
	{
		throw axis::foundation::InvalidOperationException();
	}
	return *new aapps::MultiLineCurveParser(paramList.GetParameterValue(_T("ID")).ToString());
}

const char * aafp::MultiLineCurveParserProvider::GetFeaturePath( void ) const
{
	return _featurePath.data();
}

const char * aafp::MultiLineCurveParserProvider::GetFeatureName( void ) const
{
	return "MultiLineCurveProvider";
}

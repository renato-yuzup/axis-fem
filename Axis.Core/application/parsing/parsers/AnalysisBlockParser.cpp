#include "AnalysisBlockParser.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/language/syntax/evaluation/ParameterValue.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/syntax/SkipperParser.hpp"

namespace afd = axis::foundation::definitions;
namespace asli = axis::services::language::iterators;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aslp = axis::services::language::parsing;
namespace asls = axis::services::language::syntax;
namespace asmm = axis::services::messaging;
namespace aaj = axis::application::jobs;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aafp = axis::application::factories::parsers;

aapps::AnalysisBlockParser::AnalysisBlockParser(aafp::AnalysisBlockParserProvider& parentProvider) :
_parentProvider(parentProvider)
{
	// nothing to do here
}

aapps::AnalysisBlockParser::~AnalysisBlockParser( void )
{
	// nothing to do here
}

aapps::BlockParser& aapps::AnalysisBlockParser::GetNestedContext( const axis::String& contextName, 
  const aslse::ParameterList& paramList )
{
	if (_parentProvider.ContainsProvider(contextName, paramList))
	{
		aafp::BlockProvider& provider = _parentProvider.GetProvider(contextName, paramList);
		aapps::BlockParser& nestedContext = provider.BuildParser(contextName, paramList);
		nestedContext.SetAnalysis(GetAnalysis());
		return nestedContext;
	}

	// no provider found
	throw axis::foundation::NotSupportedException();
}

aslp::ParseResult aapps::AnalysisBlockParser::Parse( const asli::InputIterator& begin, 
                                                     const asli::InputIterator& end )
{
	// this parser does not accept declarations in its context
	GetParseContext().RegisterEvent(
      asmm::ErrorMessage(0x300502, AXIS_ERROR_MSG_INVALID_DECLARATION, _T("Input parse error")));

	// "accept" line and force drop entire contents
	aslp::ParseResult result;
	result.SetResult(aslp::ParseResult::MatchOk);

	asls::SkipperParser skipper;
	result.SetLastReadPosition(skipper(begin, end));

	return result;
}

void aapps::AnalysisBlockParser::DoStartContext( void )
{
	aapc::SymbolTable& st = GetParseContext().Symbols();
  // reset step build information
	aaj::StructuralAnalysis& analysis = GetAnalysis();
	GetParseContext().SetStepOnFocus(NULL);
  GetParseContext().SetStepOnFocusIndex(-1);

	// check that this is the first settings block (only one is allowed)
	if (st.IsSymbolCurrentRoundDefined(_T("!UNIQUE!!"), aapc::SymbolTable::kAnalysisSettings))
	{	// duplicated block; trigger an error
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300570, AXIS_ERROR_MSG_DUPLICATED_BLOCK));
	}
	else
	{
		st.DefineOrRefreshSymbol(_T("!UNIQUE!!"), aapc::SymbolTable::kAnalysisSettings);
	}
}

void aapps::AnalysisBlockParser::DoCloseContext( void )
{
	// and again, reset step build information
	GetParseContext().SetStepOnFocus(NULL);
  GetParseContext().SetStepOnFocusIndex(-1);
}

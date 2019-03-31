#include "SolverParser.hpp"

#include "foundation/NotSupportedException.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"

namespace aafp  = axis::application::factories::parsers;
namespace aapc  = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace asli  = axis::services::language::iterators;
namespace aslp  = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm  = axis::services::messaging;

aapps::SolverParser::SolverParser( aafp::BlockProvider& parentProvider, 
  aapps::BlockParser& innerParser ) : _parentProvider(parentProvider), _innerParser(innerParser)
{
	// nothing to do here
}

aapps::SolverParser::~SolverParser( void )
{
	delete &_innerParser;
}

aapps::BlockParser& aapps::SolverParser::GetNestedContext( const axis::String& contextName, 
                                                           const aslse::ParameterList& paramList )
{
	// check if the parent provider knows any provider that can handle it
	if (!_parentProvider.ContainsProvider(contextName, paramList))
	{
		throw axis::foundation::NotSupportedException();
	}
	aafp::BlockProvider& subProvider = _parentProvider.GetProvider(contextName, paramList);
	return subProvider.BuildParser(contextName, paramList);
}

aslp::ParseResult aapps::SolverParser::Parse( const asli::InputIterator& begin, 
                                              const asli::InputIterator& end )
{
	// we're the lazy one, pass this task to the inner parser  :-P
	return _innerParser.Parse(begin, end);
}

void aapps::SolverParser::DoCloseContext( void )
{
	// finish parsing!
	_innerParser.CloseContext();
}

void aapps::SolverParser::DoStartContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  // initialize inner parser
	_innerParser.SetAnalysis(GetAnalysis());
	_innerParser.StartContext(GetParseContext());
	// check that this is the first settings block (only one is allowed)
	if (st.IsSymbolCurrentRoundDefined(_T("RUN_SETTINGS"), aapc::SymbolTable::kAnalysisSettings))
	{	// duplicated block; trigger an error
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300570, AXIS_ERROR_MSG_DUPLICATED_BLOCK));
	}
	else
	{
		st.DefineOrRefreshSymbol(_T("RUN_SETTINGS"), aapc::SymbolTable::kAnalysisSettings);
	}
}

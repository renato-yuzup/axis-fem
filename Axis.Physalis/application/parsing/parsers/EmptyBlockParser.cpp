#include "EmptyBlockParser.hpp"
#include "application/parsing/error_messages.hpp"
#include "services/logging/event_sources.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/syntax/SkipperParser.hpp"
#include "services/messaging/ErrorMessage.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace asls = axis::services::language::syntax;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;

aapps::EmptyBlockParser::EmptyBlockParser( aafp::BlockProvider& factory ) :
_parent(factory)
{
	// nothing to do here
}

aapps::EmptyBlockParser::~EmptyBlockParser( void )
{
	// nothing to do here
}

aapps::BlockParser& aapps::EmptyBlockParser::GetNestedContext( const axis::String& contextName, 
                                                               const aslse::ParameterList& paramList )
{
	// check if the parent provider knows any provider that can handle it
	if (!_parent.ContainsProvider(contextName, paramList))
	{
		throw axis::foundation::NotSupportedException();
	}

	aafp::BlockProvider& subProvider = _parent.GetProvider(contextName, paramList);
	return subProvider.BuildParser(contextName, paramList);
}

aslp::ParseResult aapps::EmptyBlockParser::Parse( const asli::InputIterator& begin, 
                                                  const asli::InputIterator& end )
{
	// this parser is used to not accept declarations in its context
	GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300502, AXIS_ERROR_MSG_INVALID_DECLARATION, 
                                                     _T("Input parse error")));

	// "accept" line and force drop entire contents
	aslp::ParseResult result;
	result.SetResult(aslp::ParseResult::MatchOk);

	asls::SkipperParser skipper;
	result.SetLastReadPosition(skipper(begin, end));

	return result;
}

void aapps::EmptyBlockParser::DoStartContext( void )
{
	// nothing to do here
}
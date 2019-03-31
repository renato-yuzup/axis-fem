#include "VoidParser.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/syntax/SkipperParser.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace asls = axis::services::language::syntax;
namespace aslse = axis::services::language::syntax::evaluation;

aapps::VoidParser::VoidParser( const aafp::BlockProvider& factory ) : 
_parent(factory)
{
	// nothing to do here
}

aapps::VoidParser::~VoidParser( void )
{
	// do nothing	
}

aapps::BlockParser& aapps::VoidParser::GetNestedContext( const axis::String& contextName, 
                                                         const aslse::ParameterList& paramList )
{
	if (_parent.ContainsProvider(contextName, paramList))
	{
		aafp::BlockProvider& provider = _parent.GetProvider(contextName, paramList);
		BlockParser& nestedContext = provider.BuildParser(contextName, paramList);
		nestedContext.SetAnalysis(GetAnalysis());
		return nestedContext;
	}
	// no provider found
	throw axis::foundation::NotSupportedException();
}

aslp::ParseResult aapps::VoidParser::Parse( const asli::InputIterator& begin, 
                                            const asli::InputIterator& end)
{
	// simply consume content, symbol by symbol
	aslp::ParseResult result;
	asls::SkipperParser skipper;		
	result.SetResult(aslp::ParseResult::MatchOk);	
	result.SetLastReadPosition(skipper(begin, end));
	return result;
}

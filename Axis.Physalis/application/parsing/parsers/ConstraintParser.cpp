#include "ConstraintParser.hpp"
#include "foundation/NotSupportedException.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aal = axis::application::locators;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;

aapps::ConstraintParser::ConstraintParser( aal::ConstraintParserLocator& locator ) :
_locator(locator)
{
	// nothing to do here
}

aapps::ConstraintParser::~ConstraintParser( void )
{
	// nothing to do here
}

aapps::BlockParser& aapps::ConstraintParser::GetNestedContext( const axis::String& contextName, 
                                                               const aslse::ParameterList& paramList )
{
	if (_locator.ContainsProvider(contextName, paramList))
	{
		aafp::BlockProvider& provider = _locator.GetProvider(contextName, paramList);
		BlockParser& nestedContext = provider.BuildParser(contextName, paramList);
		nestedContext.SetAnalysis(GetAnalysis());
		return nestedContext;
	}
	// no provider found
	throw axis::foundation::NotSupportedException();
}

aslp::ParseResult aapps::ConstraintParser::Parse( const asli::InputIterator& begin, 
                                                  const asli::InputIterator& end )
{
	// check if we can parse so far
	aslp::ParseResult result = _locator.TryParse(begin, end);
	if (result.IsMatch())
	{	// yes, we can -- build constraint
		return _locator.ParseAndBuild(GetAnalysis(), GetParseContext(), begin, end);
	}
	// no, we can -- return our best effort
	return result;
}

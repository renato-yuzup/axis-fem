#include "CompositeParser.hpp"
#include "AtomicExpressionParser.hpp"

namespace aslpp = axis::services::language::primitives;

aslpp::CompositeParser::CompositeParser( void )
{
	/* nothing to do here */
}

aslpp::CompositeParser::~CompositeParser( void )
{
	// delete owned objects
	_ownedExpressionObjs.ClearAndDestroy();
}

aslpp::CompositeParser& aslpp::CompositeParser::Add( const ExpressionParser& expression )
{
	_components.Add(expression);
	return *this;
}

aslpp::CompositeParser& aslpp::CompositeParser::Add( const Parser& parser )
{
	ExpressionParser& e = *new AtomicExpressionParser(parser);
	_ownedExpressionObjs.Add(e);
	return Add(e);
}

aslpp::CompositeParser& aslpp::CompositeParser::operator<<( ExpressionParser& expression )
{
	return Add(expression);
}

aslpp::CompositeParser& aslpp::CompositeParser::operator<<( const Parser& parser )
{
	return Add(parser);
}

void aslpp::CompositeParser::Remove( const ExpressionParser& expression )
{
	_components.Remove(expression);
}

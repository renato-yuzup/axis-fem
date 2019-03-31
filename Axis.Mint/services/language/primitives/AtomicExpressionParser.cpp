#include "AtomicExpressionParser.hpp"
#include "foundation/NotSupportedException.hpp"
#include "../parsing/ExpressionNode.hpp"

namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;

aslpp::AtomicExpressionParser::AtomicExpressionParser( const Parser& parser ) :
_myParser(parser)
{
	/* nothing more to do here */
}

aslpp::AtomicExpressionParser::~AtomicExpressionParser( void )
{
	/* nothing to do here */
}

aslp::ParseResult aslpp::AtomicExpressionParser::DoParse( const asli::InputIterator& begin, 
                                                          const asli::InputIterator& end, 
                                                          bool trimSpaces /*= true*/ ) const
{
	return _myParser.Parse(begin, end, trimSpaces);
}

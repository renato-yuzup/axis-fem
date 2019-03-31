#include "OrExpressionParser.hpp"

namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;

aslpp::OrExpressionParser::OrExpressionParser( void )
{
	/* nothing to do */
}

aslpp::OrExpressionParser::~OrExpressionParser( void )
{
	/* nothing to do */
}

aslp::ParseResult aslpp::OrExpressionParser::DoParse( const asli::InputIterator& begin, 
                                                      const asli::InputIterator& end, 
                                                      bool trimSpaces /*= true*/ ) const
{
	aslp::ParseResult partialOrFailedResult;
	aslp::ParseResult matchedResult;

	// initialize variables
	partialOrFailedResult.SetLastReadPosition(begin);
	bool gotPartialResult = false;
	bool gotFullMatch = false;

	// iterate through components for best match
	for (ExpressionList::Iterator it = _components.First(); it.IsValid(); ++it)
	{
		aslp::ParseResult result = it->Parse(begin, end, trimSpaces);
		if (result.IsMatch() && !gotFullMatch)
		{	// found a match -- we will use it unless we got a partial match
			matchedResult = result;	// store match and continue
			gotFullMatch = true;
		}
		if (!gotPartialResult && result.GetResult() == aslp::ParseResult::FullReadPartialMatch && 
      !gotFullMatch)
		{	// got a partial match, record it; we won't return it immediately
			// because we might run in a complete match when iterating through
			// the remaining expressions
			gotPartialResult = true;	// set flag so we capture only the first partial result
			partialOrFailedResult = result;
		}
	}

	if (gotPartialResult && gotFullMatch)
	{	// we've got partial and full match; return partial match unless the
		// full match also reached the end position
		if (matchedResult.GetLastReadPosition() == end)
		{
			return matchedResult;
		}
		else
		{
			return partialOrFailedResult;
		}
	}
	else if(gotFullMatch)
	{	// at least one full match, no partial matches
		return matchedResult;
	}
	
	// only partial match or failed
	return partialOrFailedResult;
}

#include "EnumerationParser.hpp"
#include "../parsing/EnumerationExpression.hpp"
#include "../factories/AxisGrammar.hpp"
#include "../parsing/EmptyNode.hpp"
#include "foundation/NotSupportedException.hpp"
#include "AtomicExpressionParser.hpp"

namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;

aslpp::EnumerationParser::EnumerationParser( const ExpressionParser& enumeratedExpression, bool isRequired /*= false*/ ) :
_ownedExpressionParser(false), _separatorParser(aslf::AxisGrammar::CreateOperatorParser(_T(","))), 
_isRequired(isRequired)
{
	_enumeratedExpression = &enumeratedExpression;
}

aslpp::EnumerationParser::EnumerationParser( const Parser& enumeratedExpression, bool isRequired /*= false*/ ) :
_ownedExpressionParser(true), _separatorParser(aslf::AxisGrammar::CreateOperatorParser(_T(","))), 
_isRequired(isRequired)
{
	_enumeratedExpression = new AtomicExpressionParser(enumeratedExpression);
}

aslpp::EnumerationParser::~EnumerationParser( void )
{
	if (_ownedExpressionParser)
	{
		delete _enumeratedExpression;
	}
}

aslp::ParseResult aslpp::EnumerationParser::DoParse( const asli::InputIterator& begin, 
                                                     const asli::InputIterator& end, 
                                                     bool trimSpaces /*= true*/ ) const
{
	aslp::ParseResult finalResult;
	aslp::EnumerationExpression *node = new aslp::EnumerationExpression();
	asli::InputIterator nextReadPos = begin;

	// initialize variables
	finalResult.SetLastReadPosition(begin);
	bool gotSomeResult = true;

	// first trivial case -- we have an empty string or a string filled with whitespaces
	aslpp::Parser trivialParser = aslf::AxisGrammar::CreateBlankParser();
	aslp::ParseResult trivialResult = trivialParser(begin, end);
	if (trivialResult.IsMatch() && trivialResult.GetLastReadPosition() == end)
	{	//that is, an empty string to our point of view
		if (_isRequired)
		{
			return aslp::ParseResult(aslp::ParseResult::FailedMatch, *node, 
                               trivialResult.GetLastReadPosition());		
		}
		return aslp::ParseResult(aslp::ParseResult::MatchOk, *node, 
                             trivialResult.GetLastReadPosition());		
	}
	
	// iterate through components for best match
	do
	{
		aslp::ParseResult result = _enumeratedExpression->Parse(nextReadPos, end, trimSpaces);
		if (result.IsMatch())
		{	// match ok, append to the root note
			node->AddChild(result.GetParseTree());
			nextReadPos = result.GetLastReadPosition();
			// check if we have a list separator; if not so, exit loop (end of list)
			aslp::ParseResult sepResult = _separatorParser(nextReadPos, end);
			if (sepResult.IsMatch())
			{	// ok, move on to the next loop
				nextReadPos = sepResult.GetLastReadPosition();

				// check if there is still something to parse
				if (nextReadPos == end)
				{	// we are expecting a new item but we reached the end of the
					// stream; notify it
					finalResult.SetResult(aslp::ParseResult::FullReadPartialMatch);
					gotSomeResult = false;
				}
			}
			else
			{	// end of the list, exit loop
				finalResult.SetResult(aslp::ParseResult::MatchOk);
				gotSomeResult = false;
			}
		}
		else
		{	// partial match or failed -- annotate result and exit
			finalResult.SetResult(result.GetResult());
			finalResult.SetLastReadPosition(result.GetLastReadPosition());
			gotSomeResult = false;
		}
	}while(gotSomeResult);

	if (finalResult.GetResult() == aslp::ParseResult::MatchOk)
	{	// ok, parse end -- return result
		return aslp::ParseResult(finalResult.GetResult(), *node, nextReadPos);
	}
	else
	{	// parse failed or just partial match
		delete node;
		if (finalResult.GetResult() == aslp::ParseResult::FullReadPartialMatch)
		{
			return aslp::ParseResult(finalResult.GetResult(), *new aslp::EmptyNode(), end);
		}
		else
		{
			return aslp::ParseResult(finalResult.GetResult(), *new aslp::EmptyNode(), 
                               finalResult.GetLastReadPosition());		
		}
	}
}

aslpp::CompositeParser& aslpp::EnumerationParser::Add( ExpressionParser& expression )
{
	throw axis::foundation::NotSupportedException();
}

aslpp::CompositeParser& aslpp::EnumerationParser::Add( const Parser& parser )
{
	throw axis::foundation::NotSupportedException();
}

void aslpp::EnumerationParser::Remove( const ExpressionParser& expression )
{
	throw axis::foundation::NotSupportedException();
}

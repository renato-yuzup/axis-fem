#include "GeneralExpressionParser.hpp"
#include "../parsing/EmptyNode.hpp"
#include "../parsing/ExpressionNode.hpp"
#include "../parsing/RhsExpression.hpp"

namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;

aslpp::GeneralExpressionParser::GeneralExpressionParser( void )
{
	/* nothing to do here */
}

aslpp::GeneralExpressionParser::~GeneralExpressionParser( void )
{
	/* nothing to do here */
}

aslp::ParseResult aslpp::GeneralExpressionParser::DoParse( const asli::InputIterator& begin, 
                                                           const asli::InputIterator& end, 
                                                           bool trimSpaces /*= true*/ ) const
{
	asli::InputIterator nextParsePos = begin; // position from where to start parsing
	bool atEndPosition;

	// fail if we are an empty expression
	if (_components.IsEmpty())
	{
		return aslp::ParseResult(aslp::ParseResult::FailedMatch, *new aslp::EmptyNode(), begin);
	}

	// iterate through components

	// get first finalResult, which will be the base to build the sum of all children results
	const ExpressionParser& leadingExpr = *_components.First();
	aslp::ParseResult finalResult = leadingExpr.Parse(nextParsePos, end, trimSpaces);
	if (!finalResult.IsMatch())
	{	// match failed or partial match found, return finalResult
		return finalResult;
	}

	// pointers to the leadings of parse tree
	aslp::ParseTreeNode *firstNode = &finalResult.GetParseTree();
	aslp::ParseTreeNode *lastNode = &finalResult.GetParseTree();

	// move iterator to the next parsing position
	nextParsePos = finalResult.GetLastReadPosition();

	// iterate through the rest of the children
	for (ExpressionList::Iterator it = ++_components.First(); it.IsValid(); ++it)
	{
		// if we reached the end of the input while we still have parsers to go
		// then it is a partial match -- skip loop
		atEndPosition = (nextParsePos == end);

		const ExpressionParser& expression = *it;
		aslp::ParseResult childResult = expression.Parse(nextParsePos, end, trimSpaces);

		if (!childResult.IsMatch() && atEndPosition)
		{	// we couldn't parse because we were at the end of the stream, 
			// without more chars to parse (and the next children didn't
			// accept epsilon); assume partial match and exit
			finalResult.SetResult(aslp::ParseResult::FullReadPartialMatch);
			finalResult.SetLastReadPosition(end);
			break;
		}

		// push last child result to our final result
		finalResult.SetResult(childResult.GetResult());	
		nextParsePos = childResult.GetLastReadPosition();
		finalResult.SetLastReadPosition(nextParsePos);

		if (childResult.IsMatch())
		{	// parse succeeded, append parsing node to our parsing tree; 
			// however, if it is empty, we will not add it
			if (!childResult.GetParseTree().IsEmpty())
			{
				/* Here is a special case: if the last node is empty (which
				   occurs only if the empty node was generated at the first
				   parsing operation), we must replace that node by the one
				   we have just found 
				*/
				if (lastNode->IsEmpty())
				{	// replace last node by this one we have just found
					firstNode = &childResult.GetParseTree();
					lastNode = &childResult.GetParseTree();

					// enforce reference grabbing so that the object referenced
					// by firstNode variable will be alive
					finalResult = aslp::ParseResult(finalResult.GetResult(), *firstNode, 
                                          finalResult.GetLastReadPosition());
				}
				else
				{
					lastNode->SetNextSibling(childResult.GetParseTree());
					lastNode = &childResult.GetParseTree();				
				}
			}
		}
		else
		{	// failed parsing or just a partial match -- in either cases, we don't
			// need to continue
			break;
		}
	}

	// finished parsing -- check final standings
	if (finalResult.IsMatch())
	{	// ok, parsing succeeded; append children to our node
		aslp::RhsExpression *masterNode = new aslp::RhsExpression();
		masterNode->AddChild(*firstNode);
		return aslp::ParseResult(finalResult.GetResult(), *masterNode, 
                             finalResult.GetLastReadPosition());
	}
	else
	{	// failed or partial match -- return an empty node
		return aslp::ParseResult(finalResult.GetResult(), 
						                 *new aslp::EmptyNode(), 
						                 finalResult.GetResult() == aslp::ParseResult::FullReadPartialMatch ?
						                 end : finalResult.GetLastReadPosition());
	}
}

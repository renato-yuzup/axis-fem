#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParseTreeNode.hpp"
#include "../iterators/InputIterator.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API ParseResult
{
public:
	enum Result						/* What was found when parsing operation ended */
	{
		MatchOk = 0,				/* parsing operation completed successfuly; input iterator might no be in the end, though */
		FailedMatch = 1,			/* One or more rules failed to match */
		FullReadPartialMatch = 2	/* Input ended before all parsing rule could be checked; however no error was found until then */
	};

	ParseResult(void);
	ParseResult(Result result, parsing::ParseTreeNode& parseTree, 
    const axis::services::language::iterators::InputIterator& lastPosition);
	ParseResult(const ParseResult& other);
	~ParseResult(void);

	bool IsMatch(void) const;

	Result GetResult(void) const;
	void SetResult(const Result result);

	parsing::ParseTreeNode& GetParseTree(void);
	const parsing::ParseTreeNode& GetParseTree(void) const;

	axis::services::language::iterators::InputIterator GetLastReadPosition(void) const;
	void SetLastReadPosition(const axis::services::language::iterators::InputIterator& it);

	ParseResult& operator =(const ParseResult& other);

	void ClearParseTree(void);
private:
	void Copy(const ParseResult& other);

  Result _result;
	parsing::ParseTreeNode *_parseTree;
	axis::services::language::iterators::InputIterator *_lastPosition;
};					

} } } } // axis::services::language::parsing

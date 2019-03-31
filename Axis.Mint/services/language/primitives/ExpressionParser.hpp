#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../parsing/ParseTreeNode.hpp"
#include "../parsing/ParseResult.hpp"
#include "../iterators/InputIterator.hpp"
#include "../actions/ParserAction.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API ExpressionParser
{
public:
	ExpressionParser(void);
	virtual ~ExpressionParser(void);

	axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
	axis::services::language::parsing::ParseResult operator()(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;

	ExpressionParser& AddAction(const axis::services::language::actions::ParserAction& action);
	ExpressionParser& operator <<(const axis::services::language::actions::ParserAction& action);
private:
  virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, 
    bool trimSpaces = true) const = 0;

  axis::services::language::actions::ParserAction *_action;
};		

} } } } // namespace axis::services::language::primitives

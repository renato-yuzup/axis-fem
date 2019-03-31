#pragma once
#include "CompositeParser.hpp"
#include "foundation/Axis.Mint.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API OrExpressionParser : public CompositeParser
{
public:
	OrExpressionParser(void);
	~OrExpressionParser(void);
private:
  virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
};			

} } } } // namespace axis::services::language::primitives

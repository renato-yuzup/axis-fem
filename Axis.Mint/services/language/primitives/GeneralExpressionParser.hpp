#pragma once
#include "foundation/Axis.Mint.hpp"
#include "CompositeParser.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API GeneralExpressionParser : public CompositeParser
{
protected:
	virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
public:
	GeneralExpressionParser(void);
	~GeneralExpressionParser(void);
};			

} } } } // namespace axis::services::language::primitives

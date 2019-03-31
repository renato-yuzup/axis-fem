#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ExpressionParser.hpp"
#include "Parser.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API AtomicExpressionParser : public ExpressionParser
{
public:
	AtomicExpressionParser(const Parser& parser);
	~AtomicExpressionParser(void);
private:
  virtual axis::services::language::parsing::ParseResult DoParse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true ) const;
	Parser _myParser;
};		

} } } } // namespace axis::services::language::primitives

#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../primitives/GeneralExpressionParser.hpp"

namespace axis { namespace services { namespace language { namespace syntax {

class AXISMINT_API BlockTailParser
{
public:
	BlockTailParser(void);
	~BlockTailParser(void);
	axis::String GetBlockName(void) const;
	axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	axis::services::language::parsing::ParseResult operator()(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
private:
	axis::services::language::primitives::GeneralExpressionParser _parser;
	axis::String _blockName;
};

} } } } // namespace axis::services::language::syntax

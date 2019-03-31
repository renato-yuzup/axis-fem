#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../iterators/InputIterator.hpp"
#include "evaluation/ParameterList.hpp"
#include "../parsing/EnumerationExpression.hpp"
#include "../parsing/RhsExpression.hpp"
#include "../parsing/AssignmentExpression.hpp"
#include "../parsing/IdTerminal.hpp"
#include "../parsing/NumberTerminal.hpp"
#include "../parsing/StringTerminal.hpp"
#include "../parsing/ParseResult.hpp"
#include "../primitives/OrExpressionParser.hpp"
#include "../primitives/EnumerationParser.hpp"
#include "../primitives/AssignmentParser.hpp"
#include "../primitives/GeneralExpressionParser.hpp"

namespace axis { namespace services { namespace language { namespace syntax {

class AXISMINT_API SkipperParser
{
public:
	axis::services::language::iterators::InputIterator SkipSymbol(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end) const;
	axis::services::language::iterators::InputIterator operator ()(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end) const;
};			

} } } } // namespace axis::services::language::syntax

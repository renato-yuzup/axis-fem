#pragma once
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"
#include "application/output/collectors/summarizers/SummaryType.hpp"
#include "CollectorParseResult.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class GeneralNodeCollectorParser
{
public:
  GeneralNodeCollectorParser(void);
  ~GeneralNodeCollectorParser(void);

  CollectorParseResult Parse(const axis::services::language::iterators::InputIterator& begin,
                             const axis::services::language::iterators::InputIterator& end);
private:
  void InitGrammar(void);
  CollectorParseResult InterpretParseTree(
    const axis::services::language::parsing::ParseResult& result) const;

  // our grammar rules
  axis::services::language::primitives::GeneralExpressionParser collectorStatement_;
  axis::services::language::primitives::OrExpressionParser groupingType_;
  axis::services::language::primitives::OrExpressionParser collectorType3D_;
  axis::services::language::primitives::OrExpressionParser collectorType6D_;
  axis::services::language::primitives::GeneralExpressionParser collectorType6DExpression_;
  axis::services::language::primitives::GeneralExpressionParser collectorType3DExpression_;
  axis::services::language::primitives::OrExpressionParser optionalDirection3DExpression_;
  axis::services::language::primitives::OrExpressionParser optionalDirection6DExpression_;
  axis::services::language::primitives::OrExpressionParser collectorTypeExpression_;
  axis::services::language::primitives::EnumerationParser *direction3DEnum_;
  axis::services::language::primitives::EnumerationParser *direction6DEnum_;
  axis::services::language::primitives::OrExpressionParser direction3D_;
  axis::services::language::primitives::OrExpressionParser direction6D_;
  axis::services::language::primitives::OrExpressionParser optionalSetExpression_;
  axis::services::language::primitives::GeneralExpressionParser setExpression_;
  axis::services::language::primitives::OrExpressionParser optionalScaleExpression_;
  axis::services::language::primitives::GeneralExpressionParser scaleExpression_;
  axis::services::language::primitives::OrExpressionParser anyIdentifierExpression_;
};

} } } } // namespace axis::application::factories::collectors

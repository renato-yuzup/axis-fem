#pragma once
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"
#include "application/output/collectors/summarizers/SummaryType.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

  class TextReportElementCollectorParser
  {
  public:
    class ElementCollectorParseResult
    {
    public:
      enum CollectorType
      {
        kStress,
        kStrain,
        kUndefined
      };
      ElementCollectorParseResult(const axis::services::language::parsing::ParseResult& result);
      ElementCollectorParseResult(const axis::services::language::parsing::ParseResult& result,
        CollectorType collectorType,
        const bool *directionState,
        const axis::String& targetSetName);
      ElementCollectorParseResult(const ElementCollectorParseResult& other);
      ElementCollectorParseResult& operator =(const ElementCollectorParseResult& other);

      ~ElementCollectorParseResult(void);

      axis::services::language::parsing::ParseResult GetParseResult(void) const;
      CollectorType GetCollectorType(void) const;
      axis::String GetTargetSetName(void) const;
      bool ShouldCollectDirection(int directionIndex) const;
    private:
      void Init(const bool * directionState);

      axis::services::language::parsing::ParseResult parseResult_;
      CollectorType collectorType_;
      axis::String targetSetName_;
      bool directionState_[6];
      int directionCount_;
    };

    TextReportElementCollectorParser(void);
    ~TextReportElementCollectorParser(void);

    ElementCollectorParseResult Parse(const axis::services::language::iterators::InputIterator& begin,
      const axis::services::language::iterators::InputIterator& end);
  private:
    void InitGrammar(void);
    ElementCollectorParseResult InterpretParseTree(const axis::services::language::parsing::ParseResult& result) const;

    axis::services::language::primitives::GeneralExpressionParser collectorStatement_;
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
    axis::services::language::primitives::GeneralExpressionParser setExpression_;
    axis::services::language::primitives::OrExpressionParser anyIdentifierExpression_;
  };

} } } } // namespace axis::application::factories::collectors

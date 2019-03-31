#pragma once
#include "application/output/collectors/summarizers/SummaryType.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "CollectorType.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class CollectorParseResult
{
public:
  CollectorParseResult(const axis::services::language::parsing::ParseResult& result);
  CollectorParseResult(const axis::services::language::parsing::ParseResult& result,
    CollectorType collectorType,
    axis::application::output::collectors::summarizers::SummaryType groupingType,
    const bool *directionState,
    const axis::String& targetSetName, bool actOnWholeSet,
    real scaleFactor, bool useScale);
  CollectorParseResult(const CollectorParseResult& other);
  CollectorParseResult& operator =(const CollectorParseResult& other);
  ~CollectorParseResult(void);

  axis::services::language::parsing::ParseResult GetParseResult(void) const;
  CollectorType GetCollectorType(void) const;
  axis::application::output::collectors::summarizers::SummaryType GetGroupingType(void) const;
  axis::String GetTargetSetName(void) const;
  real GetScaleFactor(void) const;
  bool DoesActOnWholeModel(void) const;
  bool ShouldCollectDirection(int directionIndex) const;
  bool ShouldScaleResults(void) const;
  int GetDirectionCount(void) const;
private:
  void Init(const bool * directionState);

  axis::services::language::parsing::ParseResult parseResult_;
  CollectorType collectorType_;
  axis::String targetSetName_;
  real scaleFactor_;
  bool actOnWholeModel_;
  bool shouldScale_;
  bool directionState_[6];
  int directionCount_;
  axis::application::output::collectors::summarizers::SummaryType groupingType_;
};

} } } } // namespace axis::application::factories::collectors

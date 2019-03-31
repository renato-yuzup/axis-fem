#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/fwd/output_collectors.hpp"
#include "EntityType.hpp"
#include "services/language/parsing/ParseResult.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class AXISPHYSALIS_API CollectorBuildResult
{
public:
  CollectorBuildResult(axis::application::output::collectors::EntityCollector *collector, 
                       EntityType collectorType,
                       const axis::services::language::parsing::ParseResult& result);
  CollectorBuildResult(axis::application::output::collectors::EntityCollector *collector, 
                       EntityType collectorType,
                       const axis::services::language::parsing::ParseResult& result,
                       bool modelIncomplete);
  ~CollectorBuildResult(void);

	axis::application::output::collectors::EntityCollector * const Collector;
	axis::services::language::parsing::ParseResult Result;
	bool IsModelIncomplete;
  const EntityType CollectorType;
};

} } } } // namespace axis::application::factories::collectors

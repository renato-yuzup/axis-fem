#pragma once
#include "GeneralSummaryNodeCollectorFactory.hpp"
#include "GeneralNodeCollectorParser.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class GeneralSummaryNodeCollectorFactory::Pimpl
{
public:
  axis::services::language::parsing::ParseResult TryParseAny( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );

  CollectorBuildResult ParseAndBuildAny(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, 
    const axis::domain::analyses::NumericalModel& model, 
    axis::application::parsing::core::ParseContext& context,
    SummaryNodeCollectorBuilder& builder);

  axis::application::output::collectors::GenericCollector& BuildCollector( 
    CollectorType collectorType, const axis::String& targetSetName, const bool * directionsToCollect,
    axis::application::output::collectors::summarizers::SummaryType summaryType,
    SummaryNodeCollectorBuilder& builder) const;

  void MarkUndefinedNodeSet(const axis::String& setName, 
    axis::application::parsing::core::ParseContext& context) const;

  GeneralNodeCollectorParser Parser;
};

} } } } // namespace axis::application::factories::collectors

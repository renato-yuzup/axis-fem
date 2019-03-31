#pragma once
#include "GeneralSummaryElementCollectorFactory.hpp"
#include "GeneralElementCollectorParser.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class GeneralSummaryElementCollectorFactory::Pimpl
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
    SummaryElementCollectorBuilder& builder);
private:
  axis::application::output::collectors::GenericCollector& BuildCollector( 
    CollectorType collectorType, const axis::String& targetSetName, const bool * directionsToCollect,
    axis::application::output::collectors::summarizers::SummaryType summaryType,
    SummaryElementCollectorBuilder& builder) const;

  void MarkUndefinedElementSet(const axis::String& setName, 
    axis::application::parsing::core::ParseContext& context) const;

  GeneralElementCollectorParser parser_;
};

} } } } // namespace axis::application::factories::collectors

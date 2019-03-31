#pragma once
#include "TextReportElementCollectorFactory.hpp"
#include "TextReportElementCollectorParser.hpp"
#include "application/output/collectors/GenericCollector.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

  class TextReportElementCollectorFactory::Pimpl
  {
  public:
    axis::services::language::parsing::ParseResult TryParseAny( 
      const axis::services::language::iterators::InputIterator& begin, 
      const axis::services::language::iterators::InputIterator& end );

    CollectorBuildResult ParseAndBuildAny(
      const axis::services::language::iterators::InputIterator& begin, 
      const axis::services::language::iterators::InputIterator& end, 
      const axis::domain::analyses::NumericalModel& model, 
      axis::application::parsing::core::ParseContext& context);

    axis::application::output::collectors::GenericCollector& BuildCollector( 
      TextReportElementCollectorParser::ElementCollectorParseResult::CollectorType collectorType,
      const axis::String& targetSetName, 
      const bool * directionsToCollect) const;

    void MarkUndefinedElementSet(const axis::String& setName, 
      axis::application::parsing::core::ParseContext& context) const;

    TextReportElementCollectorParser Parser;
  };

} } } } // namespace axis::application::factories::collectors

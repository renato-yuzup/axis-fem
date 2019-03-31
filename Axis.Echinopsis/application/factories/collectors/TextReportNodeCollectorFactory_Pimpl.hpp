#pragma once
#include "TextReportNodeCollectorFactory.hpp"
#include "TextReportNodeCollectorParser.hpp"
#include "application/output/collectors/GenericCollector.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

  class TextReportNodeCollectorFactory::Pimpl
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
      TextReportNodeCollectorParser::NodeCollectorParseResult::CollectorType collectorType,
      const axis::String& targetSetName, 
      const bool * directionsToCollect) const;

    void MarkUndefinedNodeSet(const axis::String& setName, 
      axis::application::parsing::core::ParseContext& context) const;

    TextReportNodeCollectorParser Parser;
  };

} } } } // namespace axis::application::factories::collectors

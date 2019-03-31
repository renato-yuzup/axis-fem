#pragma once
#include "GeneralElementCollectorFactory.hpp"
#include "GeneralElementCollectorParser.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class GeneralElementCollectorFactory::Pimpl
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
    ElementCollectorBuilder& builder);
private:
  axis::application::output::collectors::ElementSetCollector& BuildCollector( 
    axis::application::factories::collectors::CollectorType collectorType,
    const axis::String& targetSetName, 
    const bool * directionsToCollect,
    ElementCollectorBuilder& builder) const;

  void MarkUndefinedElementSet(const axis::String& setName, 
    axis::application::parsing::core::ParseContext& context) const;

  GeneralElementCollectorParser parser_;
};

} } } } // namespace axis::application::factories::collectors

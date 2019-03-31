#pragma once
#include "GeneralNodeCollectorFactory.hpp"
#include "GeneralNodeCollectorParser.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class GeneralNodeCollectorFactory::Pimpl
{
public:
  axis::services::language::parsing::ParseResult TryParseAny( 
            const axis::services::language::iterators::InputIterator& begin, 
            const axis::services::language::iterators::InputIterator& end );

  CollectorBuildResult  ParseAndBuildAny(
            const axis::services::language::iterators::InputIterator& begin, 
            const axis::services::language::iterators::InputIterator& end, 
            const axis::domain::analyses::NumericalModel& model, 
            axis::application::parsing::core::ParseContext& context,
            NodeCollectorBuilder& builder);

  axis::application::output::collectors::NodeSetCollector& BuildCollector( 
            CollectorType collectorType,
            const axis::String& targetSetName, 
            const bool * directionsToCollect,
            NodeCollectorBuilder& builder) const;

  void MarkUndefinedNodeSet(const axis::String& setName, 
                            axis::application::parsing::core::ParseContext& context) const;

  GeneralNodeCollectorParser Parser;
};

} } } } // namespace axis::application::factories::collectors

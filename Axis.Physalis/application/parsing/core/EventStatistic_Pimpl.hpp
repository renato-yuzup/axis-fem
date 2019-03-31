#pragma once
#include "ParseContext.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class ParseContext::EventStatistic::Pimpl
{
public:
  long errorEventCount;
  long warningEventCount;
  long infoEventCount;
  axis::services::messaging::EventMessage *lastEvent;
};

} }	} } // namespace axis::application::parsing::core


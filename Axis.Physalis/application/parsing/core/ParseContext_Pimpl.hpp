#pragma once
#include "ParseContext.hpp"
#include "AxisString.hpp"
#include "EventStatistic.hpp"
#include "EntityLabeler.hpp"
#include "Sketchbook.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class ParseContext::Pimpl
{
public:
  EventStatistic eventStatistics;
  SymbolTable *symbols;
  EntityLabeler labeler;
  Sketchbook *sketchbook;
};

} }	} } // namespace axis::application::parsing::core

#pragma once
#include "ParseContextConcrete.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class ParseContextConcrete::Pimpl
{
public:
  axis::String parseSourceName;
  unsigned long parseSourceCursorLocation;
  ParseContext::RunMode runMode;
  int roundIndex;
  axis::application::jobs::AnalysisStep *stepOnFocus;
  int stepOnFocusIndex;
};

} } } } // namespace axis::application::parsing::core

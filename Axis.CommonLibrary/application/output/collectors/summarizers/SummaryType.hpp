#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace summarizers {

/**
  * Defines how data will be grouped by a summary collector.
  */
enum AXISCOMMONLIBRARY_API SummaryType
{
  kNone,
  kAverage,
  kMaximum,
  kMinimum,
  kSum
};

} } } } } // namespace axis::application::output::collectors::summarizers

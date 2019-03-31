#pragma once
#include "foundation/Axis.Capsicum.hpp"

namespace axis { namespace application { namespace scheduling { namespace scheduler {

/**
 * Defines type of selected processing resource.
 */
enum class AXISCAPSICUM_API SchedulerType
{
  ///< Any processing resource.
  kAnyScheduler,
  ///< Process in local CPU.
  kCPUScheduler,
  ///< Process in local GPU.
  kGPUScheduler,
  ///< Processing resource not determined or unknown.
  kUndefinedScheduler
};

} } } } // namespace axis::application::scheduling::scheduler

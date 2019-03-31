#include "SchedulingDecision.hpp"
#include "SchedulingDecision.hpp"

namespace aasb = axis::application::scheduling::broker;
namespace aass = axis::application::scheduling::scheduler;

aasb::SchedulingDecision::SchedulingDecision( void )
{
  Enforced = false;
  TargetResource = aass::SchedulerType::kUndefinedScheduler;
  Fallback = false;
  Error = false;
  AvailableResources = true;
}

aasb::SchedulingDecision::~SchedulingDecision( void )
{
  // nothing to do here
}

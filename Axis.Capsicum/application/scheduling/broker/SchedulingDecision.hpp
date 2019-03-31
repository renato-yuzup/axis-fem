#pragma once
#include "application/scheduling/scheduler/SchedulerType.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace broker {

/**
 * Represents a decision made by the scheduler about where the active processing 
 * request should be run.
 */
class SchedulingDecision
{
public:
  SchedulingDecision(void);
  ~SchedulingDecision(void);

  bool Enforced;
  axis::application::scheduling::scheduler::SchedulerType TargetResource;
  bool Fallback;
  bool Error;
  bool AvailableResources;
};

} } } } // namespace axis::application::scheduling::broker

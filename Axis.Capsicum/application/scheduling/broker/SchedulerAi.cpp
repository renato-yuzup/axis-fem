#include "SchedulerAi.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aasb = axis::application::scheduling::broker;
namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;

aasb::SchedulerAi::SchedulerAi(void)
{
  // nothing to do here
}

aasb::SchedulerAi::~SchedulerAi(void)
{
  // nothing to do here
}

bool aasb::SchedulerAi::ShouldRunOnGPU( const ada::NumericalModel& model, 
                                        const adal::Solver& solver )
{
  size_type elementCount = model.Elements().Count();
  
  // for now, only element count matters; a more complex algorithm can be put here
  // to determine usability of GPU processing
  return elementCount >= 10000;
}

bool aasb::SchedulerAi::EvaluateGPUQueue( const QueueStatistics& statistics, 
                                         const ada::NumericalModel& model, 
                                         const adal::Solver& solver )
{
  // TODO: Correct queue occupancy evaluation should be performed
  return true;
}

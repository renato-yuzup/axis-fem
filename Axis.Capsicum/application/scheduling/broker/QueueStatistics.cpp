#include "QueueStatistics.hpp"
#include "QueueStatistics.hpp"

namespace aasb = axis::application::scheduling::broker;

aasb::QueueStatistics::QueueStatistics(void) :
  CPUQueueWeight(0), GPUQueueWeight(0)
{
  // nothing to do here
}

aasb::QueueStatistics::~QueueStatistics(void)
{
  // nothing to do here
}

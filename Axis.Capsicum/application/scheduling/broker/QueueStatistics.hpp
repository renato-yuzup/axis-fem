#pragma once

namespace axis { namespace application { namespace scheduling { namespace broker {

/**
 * Contains statistics about processing queue occupancy.
 */
class QueueStatistics
{
public:
  QueueStatistics(void);
  ~QueueStatistics(void);

  double CPUQueueWeight;
  double GPUQueueWeight;
};

} } } } // namespace axis::application::scheduling::broker

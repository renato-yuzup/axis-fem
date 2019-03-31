#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "application/scheduling/scheduler/ExecutionRequest.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace dispatchers {

/**
 * Provides services to forward and start processing jobs in local CPU.
 */
class CPUDispatcher : public axis::services::messaging::CollectorHub
{
public:
  CPUDispatcher(void);
  ~CPUDispatcher(void);

  /**
   * Dispatches a job for execution.
   *
   * @param [in,out] jobRequest The job request.
   */
  void DispatchJob(
    axis::application::scheduling::scheduler::ExecutionRequest& jobRequest);
};

} } } } // namespace axis::application::scheduling::dispatchers

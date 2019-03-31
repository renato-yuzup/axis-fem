#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "application/scheduling/dispatchers/CPUDispatcher.hpp"
#include "application/scheduling/dispatchers/GPUDispatcher.hpp"
#include "QueueStatistics.hpp"
#include "application/scheduling/scheduler/ExecutionRequest.hpp"
#include "foundation/computing/ResourceManager.hpp"
#include "nocopy.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace broker {

/**
 * Receives processing requests, queues and start processing tasks.
**/
class ProcessingQueueBroker : public axis::services::messaging::CollectorHub
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] manager The object that manages processing resources in
   *                 the system.
   */
  ProcessingQueueBroker(axis::foundation::computing::ResourceManager& manager);
  ~ProcessingQueueBroker(void);

  /**
   * Returns statistics about how occupied is the processing queues.
   *
   * @return Queue statistics.
   */
  QueueStatistics EvaluateQueueOccupancy(void) const;

  /**
   * Queues job processing task for GPU.
   *
   * @param [in,out] jobRequest The job request.
   */
  void QueueGPUJob(
    axis::application::scheduling::scheduler::ExecutionRequest& jobRequest);

  /**
   * Queues job processing task for CPU.
   *
   * @param [in,out] jobRequest The job request.
   */
  void QueueCPUJob(
    axis::application::scheduling::scheduler::ExecutionRequest& jobRequest);
private:
  axis::application::scheduling::dispatchers::CPUDispatcher cpuDispatcher_;
  axis::application::scheduling::dispatchers::GPUDispatcher gpuDispatcher_;
  axis::foundation::computing::ResourceManager& manager_;

  DISALLOW_COPY_AND_ASSIGN(ProcessingQueueBroker);
};

} } } } // namespace axis::application::scheduling::broker

#include "ProcessingQueueBroker.hpp"

namespace aasb = axis::application::scheduling::broker;
namespace aass = axis::application::scheduling::scheduler;
namespace afc = axis::foundation::computing;

aasb::ProcessingQueueBroker::ProcessingQueueBroker(afc::ResourceManager& manager) : 
  manager_(manager)
{
  // nothing to do here
}


aasb::ProcessingQueueBroker::~ProcessingQueueBroker(void)
{
  // nothing to do here
}

aasb::QueueStatistics aasb::ProcessingQueueBroker::EvaluateQueueOccupancy( void ) const
{
  // TODO: Queue occupancy correct calculation should be implemented
  return QueueStatistics();
}

void aasb::ProcessingQueueBroker::QueueGPUJob( aass::ExecutionRequest& jobRequest )
{
  gpuDispatcher_.ConnectListener(*this);
  gpuDispatcher_.DispatchJob(jobRequest, manager_);
  gpuDispatcher_.DisconnectListener(*this);
}

void aasb::ProcessingQueueBroker::QueueCPUJob( aass::ExecutionRequest& jobRequest )
{
  cpuDispatcher_.ConnectListener(*this);
  cpuDispatcher_.DispatchJob(jobRequest);
  cpuDispatcher_.DisconnectListener(*this);
}

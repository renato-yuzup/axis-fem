#include "ExecutionScheduler.hpp"
#include <iostream>
#include "ExecutionScheduler_Pimpl.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "foundation/capsicum_error.hpp"
#include "foundation/capsicum_info.hpp"

namespace ada  = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace aasb = axis::application::scheduling::broker;
namespace aass = axis::application::scheduling::scheduler;
namespace asmm = axis::services::messaging;

aass::ExecutionScheduler * aass::ExecutionScheduler::scheduler_ = nullptr;

aass::ExecutionScheduler::ExecutionScheduler(void)
{
  pimpl_ = new Pimpl(*this);
}

aass::ExecutionScheduler::~ExecutionScheduler(void)
{
  delete pimpl_;
  pimpl_ = nullptr;
}

bool aass::ExecutionScheduler::Submit( ExecutionRequest& request )
{
  JobInspector& inspector = *pimpl_->Inspector;
  aasb::ProcessingQueueBroker &queueBroker = *pimpl_->QueueBroker;
  aasb::SchedulerBroker& schedulingBroker = *pimpl_->JobBroker;

  pimpl_->Notify(AXIS_INFO_ID_SCHEDULING_ACTIVE, AXIS_INFO_MSG_SCHEDULING_ACTIVE);

  // scan processing devices
  pimpl_->Manager->Rescan();

  // inspect job to see it is not malformed
  bool isJobOk = 
    inspector.Inspect(request.GetNumericalModel(), request.GetSolver());
  if (!isJobOk)
  { 
    pimpl_->Error(AXIS_ERROR_ID_SCHEDULING_MALFORMED_JOB, 
      AXIS_ERROR_MSG_SCHEDULING_MALFORMED_JOB, request.GetJobName());
    return false;
  }
  pimpl_->Notify(AXIS_INFO_ID_SCHEDULING_SUBMISSION_OK, 
    AXIS_INFO_MSG_SCHEDULING_SUBMISSION_OK, request.GetJobName());

  // check which processing queue is better suited
  SchedulerType requestedResource = inspector.GetJobRequestedResource();
  schedulingBroker.ConnectListener(*this);
  aasb::SchedulingDecision decision = schedulingBroker.DecideTarget(
    request.GetNumericalModel(), request.GetSolver(), requestedResource, 
    queueBroker);
  schedulingBroker.DisconnectListener(*this);  
  queueBroker.ConnectListener(*this);
  if (decision.Error)
  {
    std::wcout << _T("ERROR! Failed to decide target processing unit!") << std::endl;
    pimpl_->Error(AXIS_ERROR_ID_SCHEDULING_FAILED, 
      AXIS_ERROR_MSG_SCHEDULING_FAILED, request.GetJobName());
    pimpl_->Notify(AXIS_INFO_ID_SCHEDULING_FINISHED, 
      AXIS_INFO_MSG_SCHEDULING_FINISHED);
    return false;
  }

  bool queued = true;
  // forward to adequate processing queue
  switch (decision.TargetResource)
  {
  case SchedulerType::kCPUScheduler:
    std::wcout << _T("INFO: Job queued for CPU.") << std::endl;
    queueBroker.QueueCPUJob(request);
    break;
  case SchedulerType::kGPUScheduler:
    std::wcout << _T("INFO: Available GPUs found: ") << pimpl_->Manager->GetAvailableGPUCount() << std::endl;
    std::wcout << _T("INFO: Job queued for GPU.") << std::endl;
    queueBroker.QueueGPUJob(request);
    break;
  default: // undefined queue type
    std::wcout << _T("ERROR! Job scheduling failed!") << std::endl;
    pimpl_->Error(AXIS_ERROR_ID_SCHEDULING_FAILED, 
      AXIS_ERROR_MSG_SCHEDULING_FAILED, request.GetJobName());
    queued = false;
    break;
  }

  pimpl_->Notify(AXIS_INFO_ID_SCHEDULING_FINISHED, 
    AXIS_INFO_MSG_SCHEDULING_FINISHED);
  queueBroker.DisconnectListener(*this);
  return queued;
}

aass::ExecutionScheduler& aass::ExecutionScheduler::GetActive( void )
{
  if (scheduler_ == nullptr)
  {
    scheduler_ = new aass::ExecutionScheduler();
  }
  return *scheduler_;
}

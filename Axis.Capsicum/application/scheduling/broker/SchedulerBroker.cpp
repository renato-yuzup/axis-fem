#include "SchedulerBroker.hpp"
#include <iostream>
#include "SchedulerAi.hpp"
#include "foundation/computing/ResourceManager.hpp"
#include "ProcessingQueueBroker.hpp"
#include "System.hpp"
#include "domain/algorithms/Solver.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/collections/ElementSet.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "foundation/capsicum_error.hpp"
#include "foundation/capsicum_info.hpp"
#include "foundation/capsicum_warning.hpp"

namespace ada  = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace adc  = axis::domain::collections;
namespace ade  = axis::domain::elements;
namespace aasb = axis::application::scheduling::broker;
namespace aass = axis::application::scheduling::scheduler;
namespace afc  = axis::foundation::computing;
namespace asmm = axis::services::messaging;

aasb::SchedulerBroker::SchedulerBroker(afc::ResourceManager& manager) : manager_(manager)
{
  // nothing to do here
}

aasb::SchedulerBroker::~SchedulerBroker(void)
{
  delete &manager_;
}

aasb::SchedulingDecision aasb::SchedulerBroker::DecideTarget(const ada::NumericalModel& model, 
                                                           const adal::Solver& solver,
                                                           aass::SchedulerType requestedResourceType,
                                                           const ProcessingQueueBroker& broker)
{
  SchedulerAi ai;
  SchedulingDecision decision;

  // check if resource request clashes with environment configuration 
  if (!DoesJobFulfillEnvironmentRequirements(requestedResourceType))
  {
    std::wcerr << _T("ERROR! Job does not fulfill environment requirements!") << std::endl;
    FailScheduling(ClashReason::kDeniedByEnvironment);
    decision.TargetResource = aass::SchedulerType::kUndefinedScheduler;
    decision.Error = true;
    return decision;
  }
  
  bool runOnGPU = (requestedResourceType == aass::SchedulerType::kGPUScheduler || 
    requestedResourceType == aass::SchedulerType::kAnyScheduler);
  bool runOnCPU = (requestedResourceType == aass::SchedulerType::kCPUScheduler || 
    requestedResourceType == aass::SchedulerType::kAnyScheduler);

  bool enforced = EnforceEnvironmentRestrictions(runOnCPU, runOnGPU);
  decision.Enforced = enforced;
  if (enforced)
  {
    Notify(AXIS_INFO_ID_SCHEDULING_ENFORCEMENT_APPLIED, AXIS_INFO_MSG_SCHEDULING_ENFORCEMENT_APPLIED);
  }
  else
  {
    // evaluate problem if it is worth running on GPU
    bool shouldUseGPU = ai.ShouldRunOnGPU(model, solver);

    // test for fallback conditions
    if (shouldUseGPU && runOnGPU)
    { // try to use GPU
      if (!manager_.IsGPUResourcesAvailable())
      { // there is no supported GPU installed 
        decision.Fallback = true;
        runOnGPU = false;
        Warn(AXIS_WARN_ID_SCHEDULING_NO_GPU_FOUND, AXIS_WARN_MSG_SCHEDULING_NO_GPU_FOUND);
      }
      else 
      {
        QueueStatistics stats = broker.EvaluateQueueOccupancy();
        bool gpuIsBetter = ai.EvaluateGPUQueue(stats, model, solver);
        decision.Fallback = !gpuIsBetter;
        runOnGPU = gpuIsBetter;

        // however, we can only fallback if the job supports it
        if (!runOnCPU)
        { // we can't fallback because the job does not allow it; just let it be...
          runOnGPU = true;
          decision.Fallback = false;
        }
        if (decision.Fallback)
        {
          Warn(AXIS_WARN_ID_SCHEDULING_GPU_BUSY, AXIS_WARN_MSG_SCHEDULING_GPU_BUSY);
        }
      }
    }
    else if (!shouldUseGPU && runOnGPU)
    {
      Notify(AXIS_INFO_ID_SCHEDULING_SIMPLE_JOB_FOR_GPU, AXIS_INFO_MSG_SCHEDULING_SIMPLE_JOB_FOR_GPU);
    }
  }

  decision.TargetResource = runOnGPU? aass::SchedulerType::kGPUScheduler : aass::SchedulerType::kCPUScheduler;
  
  // check for resources availability
  if ((runOnGPU && !manager_.IsGPUResourcesAvailable()) || 
      (runOnCPU && !manager_.IsCPUResourcesAvailable()))
  { // there is no place to run this job!
    decision.Error = true;
    decision.AvailableResources = false;
    decision.TargetResource = aass::SchedulerType::kUndefinedScheduler;
    FailScheduling(ClashReason::kNoResourcesAvailable);
    std::wcerr << _T("ERROR! Job requires GPU, but no device resources are available!") << std::endl;
  }

  return decision;
}

aass::SchedulerType aasb::SchedulerBroker::ResolveMandatoryResource( void ) const
{
  ApplicationEnvironment& env = System::Environment();
  if (env.ForceCPU())
  {
    return aass::SchedulerType::kCPUScheduler;
  }
  else if (env.ForceGPU())
  {
    return aass::SchedulerType::kGPUScheduler;
  }

  // if both flags are set of none of them, any resource is fine
  return aass::SchedulerType::kAnyScheduler;
}

bool aasb::SchedulerBroker::DoesJobFulfillEnvironmentRequirements( aass::SchedulerType jobDemandedResource ) const
{
  aass::SchedulerType mandatoryResource = ResolveMandatoryResource();
  // By using Disjunctive Normal Form, we can resume fulfillment to the following statement:
  return !(
    ((mandatoryResource == aass::SchedulerType::kGPUScheduler && jobDemandedResource == aass::SchedulerType::kCPUScheduler)) ||
    ((mandatoryResource == aass::SchedulerType::kCPUScheduler && jobDemandedResource == aass::SchedulerType::kGPUScheduler))
  );
}

bool aasb::SchedulerBroker::EnforceEnvironmentRestrictions( bool& cpuCapability, bool& gpuCapability )
{
  aass::SchedulerType mandatoryResource = ResolveMandatoryResource();
  bool newCPUCaps, newGPUCaps;
  newCPUCaps = (mandatoryResource == aass::SchedulerType::kAnyScheduler || 
                mandatoryResource == aass::SchedulerType::kCPUScheduler) && cpuCapability;
  newGPUCaps = (mandatoryResource == aass::SchedulerType::kAnyScheduler || 
                mandatoryResource == aass::SchedulerType::kGPUScheduler) && gpuCapability;
  bool enforced = (newCPUCaps != cpuCapability) || (newGPUCaps != gpuCapability);
  cpuCapability = newCPUCaps; 
  gpuCapability = newGPUCaps;
  return enforced;
}

void aasb::SchedulerBroker::FailScheduling( const ClashReason& clashReason )
{
  switch (clashReason)
  {
  case ClashReason::kDeniedByEnvironment:
    Error(AXIS_ERROR_ID_SCHEDULING_CLASH, AXIS_ERROR_MSG_SCHEDULING_CLASH);
    break;
  case ClashReason::kNoResourcesAvailable:
    Error(AXIS_ERROR_ID_SCHEDULING_UNAVAILABLE_RESOURCE, AXIS_ERROR_MSG_SCHEDULING_UNAVAILABLE_RESOURCE);
    break;
  }
}

void aasb::SchedulerBroker::Error( asmm::Message::id_type messageId, const axis::String& message )
{
  DispatchMessage(asmm::ErrorMessage(messageId, message));
}

void aasb::SchedulerBroker::Error( asmm::Message::id_type messageId, const axis::String& message, const axis::String& str1 )
{
  String s = message;
  s.replace(_T("%1"), str1);
  DispatchMessage(asmm::ErrorMessage(messageId, s));
}

void aasb::SchedulerBroker::Notify( asmm::Message::id_type messageId, const axis::String& message )
{
  DispatchMessage(asmm::InfoMessage(messageId, message));
}

void aasb::SchedulerBroker::Notify( asmm::Message::id_type messageId, const axis::String& message, const axis::String& str1 )
{
  String s = message;
  s.replace(_T("%1"), str1);
  DispatchMessage(asmm::InfoMessage(messageId, s));
}

void aasb::SchedulerBroker::Warn( asmm::Message::id_type messageId, const axis::String& message )
{
  DispatchMessage(asmm::WarningMessage(messageId, message));
}

void aasb::SchedulerBroker::Warn( asmm::Message::id_type messageId, const axis::String& message, const axis::String& str1 )
{
  String s = message;
  s.replace(_T("%1"), str1);
  DispatchMessage(asmm::WarningMessage(messageId, s));
}

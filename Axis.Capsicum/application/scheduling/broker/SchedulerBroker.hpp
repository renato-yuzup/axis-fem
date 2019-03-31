#pragma once
#include "domain/fwd/numerical_model.hpp"
#include "domain/fwd/solver_fwd.hpp"
#include "application/scheduling/scheduler/SchedulerType.hpp"
#include "ClashReason.hpp"
#include "foundation/computing/ResourceManager.hpp"
#include "SchedulingDecision.hpp"
#include "services/messaging/CollectorEndpoint.hpp"
#include "nocopy.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace broker {

class ProcessingQueueBroker;

/**
 * Provides service to analyze and forward a job to the most appropriate 
 * processing queue.
**/
class SchedulerBroker : public axis::services::messaging::CollectorEndpoint
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] manager The object that manages processing resources in
   *                 the system.
   */
  SchedulerBroker(axis::foundation::computing::ResourceManager& manager);
  ~SchedulerBroker(void);

  /**
   * Decides upon which processing resource should the scheduler use.
   *
   * @param model                 Numerical model of the current job.
   * @param solver                Solver of the current job.
   * @param requestedResourceType Type of the job requested resource.
   * @param queueBroker           The queue broker.
   *
   * @return The scheduling decision.
  **/
  SchedulingDecision DecideTarget(
    const axis::domain::analyses::NumericalModel& model, 
    const axis::domain::algorithms::Solver& solver,
    axis::application::scheduling::scheduler::SchedulerType requestedResourceType,
    const ProcessingQueueBroker& queueBroker);
private:
  /**
   * Resolves which resource type is mandatory for scheduling, as defined in
   * system environment settings.
   *
   * @return The mandatory resource type.
  **/
  axis::application::scheduling::scheduler::SchedulerType 
    ResolveMandatoryResource(void) const;

  /**
   * Returns if current job can be accepted according to environment scheduling 
   * settings.
   *
   * @param jobDemandedResource The job demanded resource type.
   *
   * @return true if it can be accepted, false otherwise.
  **/
  bool DoesJobFulfillEnvironmentRequirements(
    axis::application::scheduling::scheduler::SchedulerType jobDemandedResource
    ) const;

  /**
   * Enforces environment restrictions on job capabilities.
   *
   * @param [in,out] cpuCapability The job CPU capability.
   * @param [in,out] gpuCapability The job GPU capability.
   *
   * @return true if it capabilities were changed to attend environment 
   *         restrictions, false otherwise.
  **/
  bool EnforceEnvironmentRestrictions(bool& cpuCapability, bool& gpuCapability);

  /**
   * Executes procedures if scheduling could not be performed.
   *
   * @param [in,out] clashReason The clash reason, which is the cause of the 
   *                 failure.
  **/
  void FailScheduling(const ClashReason& clashReason);

  void Error(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message);
  void Error(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message, const axis::String& str1);
  void Notify(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message);
  void Notify(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message, const axis::String& str1);
  void Warn(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message);
  void Warn(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message, const axis::String& str1);

  axis::foundation::computing::ResourceManager& manager_;

  DISALLOW_COPY_AND_ASSIGN(SchedulerBroker);
};

} } } } // namespace axis::application::scheduling::broker

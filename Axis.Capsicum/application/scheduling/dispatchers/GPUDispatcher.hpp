#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "application/scheduling/scheduler/ExecutionRequest.hpp"
#include "foundation/computing/ResourceManager.hpp"
#include "foundation/fwd/memory_fwd.hpp"
#include "application/executors/gpu/facades/GPUModelFacade.hpp"
#include "application/executors/gpu/facades/GPUSolverFacade.hpp"

namespace axis { namespace services { namespace scheduling { namespace gpu {
class KernelScheduler;
class GPUSink;
} } } } // namespace axis::services::scheduling::gpu

namespace axis { namespace application { namespace scheduling { 
  namespace dispatchers {

class DataCoalescer;

/**
 * Provides services to forward and start processing jobs in local GPU.
 */
class GPUDispatcher : public axis::services::messaging::CollectorHub
{
public:
  GPUDispatcher(void);
  ~GPUDispatcher(void);

  /**
   * Dispatches a job for execution.
   *
   * @param [in,out] jobRequest The job request.
   */
  void DispatchJob(
    axis::application::scheduling::scheduler::ExecutionRequest& jobRequest,
    axis::foundation::computing::ResourceManager& manager);
private:
  void BuildModelFacade(
    axis::application::executors::gpu::facades::GPUModelFacade& facade,
    axis::application::scheduling::dispatchers::DataCoalescer& coalescer,
    axis::services::scheduling::gpu::KernelScheduler& scheduler,
    axis::services::scheduling::gpu::GPUSink& errSink);
  void PrepareNumericalModel(axis::domain::analyses::NumericalModel& model);
  void SetUpSolverFacade(
    axis::application::executors::gpu::facades::GPUSolverFacade& solverFacade,
    axis::application::scheduling::dispatchers::DataCoalescer& coalescer,
    axis::services::scheduling::gpu::KernelScheduler& scheduler,
    axis::domain::algorithms::Solver& solver, 
    axis::domain::analyses::NumericalModel& model, 
    axis::domain::analyses::AnalysisTimeline& timeline,
    axis::services::scheduling::gpu::GPUSink& errSink);
};

} } } } // namespace axis::application::scheduling::dispatchers

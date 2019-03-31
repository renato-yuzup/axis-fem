#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "application/executors/gpu/facades/GPUModelFacade.hpp"
#include "application/executors/gpu/facades/GPUSolverFacade.hpp"
#include "domain/algorithms/Solver.hpp"
#include "domain/fwd/solver_fwd.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace dispatchers {

/**
 * Implements similar logic to FiniteElementAnalysis object but directed to
 * GPU processing.
 * @sa FiniteElementAnalysis
 */
class GPUFiniteElementAnalysis : public axis::services::messaging::CollectorHub
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] gpuSolver         The GPU-capable solver.
   * @param [in,out] solverFacade      The solver facade object.
   * @param [in,out] gpuNumericalModel Pointer to the reduced numerical model.
   * @param [in,out] modelFacade       The model operator facade.
   */
  GPUFiniteElementAnalysis(axis::domain::algorithms::Solver& gpuSolver, 
    axis::application::executors::gpu::facades::GPUSolverFacade& solverFacade,
    axis::foundation::memory::RelativePointer& gpuNumericalModel,
    axis::application::executors::gpu::facades::GPUModelFacade& modelFacade);
  ~GPUFiniteElementAnalysis(void);

  /**
   * Runs analysis.
   *
   * @param [in,out] timeline The analysis timeline.
   */
  void Run(axis::domain::analyses::AnalysisTimeline& timeline);
private:
  axis::domain::algorithms::Solver& gpuSolver_; 
  axis::foundation::memory::RelativePointer& gpuNumericalModel_;
  axis::application::executors::gpu::facades::GPUSolverFacade& solverFacade_;
  axis::application::executors::gpu::facades::GPUModelFacade& modelFacade_;
};

} } } } // namespace axis::application::scheduling::dispatchers

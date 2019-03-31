#pragma once
#include "services/memory/commands/MemoryCommand.hpp"
#include "domain/algorithms/Solver.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands {

/**
 * Executes initialization procedures for the active solver object. Because 
 * it is expected that initialization occurs in the host, execution must be
 * done prior to mirroring memory to external processing device.
 */
class SolverInitializerCommand : 
  public axis::services::memory::commands::MemoryCommand
{
public:
  SolverInitializerCommand(axis::domain::algorithms::Solver& solver, 
                           axis::domain::analyses::NumericalModel& model, 
                           axis::domain::analyses::AnalysisTimeline& timeline);
  ~SolverInitializerCommand(void);
  virtual void Execute( void *baseAddress, void *gpuBaseAddress );
  virtual MemoryCommand& Clone( void ) const;
private:
  axis::domain::algorithms::Solver& solver_;
  axis::domain::analyses::NumericalModel& model_;
  axis::domain::analyses::AnalysisTimeline& timeline_;
};

} } } } } // namespace axis::application::executors::gpu::commands

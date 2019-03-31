#include "SolverInitializerCommand.hpp"

namespace adal  = axis::domain::algorithms;
namespace ada   = axis::domain::analyses;
namespace aaegc = axis::application::executors::gpu::commands;
namespace asmc  = axis::services::memory::commands;
namespace afm   = axis::foundation::memory;

aaegc::SolverInitializerCommand::SolverInitializerCommand(adal::Solver& solver,
  ada::NumericalModel& model, ada::AnalysisTimeline& timeline) :
solver_(solver), model_(model), timeline_(timeline)
{
  // nothing to do here
}

aaegc::SolverInitializerCommand::~SolverInitializerCommand(void)
{
  // nothing to do here
}

void aaegc::SolverInitializerCommand::Execute( void *baseAddress, void * )
{
  // request solver to initialize data
  solver_.PrepareGPUData(model_, timeline_);
}

asmc::MemoryCommand& aaegc::SolverInitializerCommand::Clone( void ) const
{
  return *new SolverInitializerCommand(solver_, model_, timeline_);
}

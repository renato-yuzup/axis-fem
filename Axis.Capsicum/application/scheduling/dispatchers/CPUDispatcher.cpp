#include "CPUDispatcher.hpp"
#include "domain/algorithms/Solver.hpp"

namespace aass = axis::application::scheduling::scheduler;
namespace aasd = axis::application::scheduling::dispatchers;
namespace ada  = axis::domain::analyses;
namespace adal = axis::domain::algorithms;

aasd::CPUDispatcher::CPUDispatcher( void )
{
  // nothing to do here
}

aasd::CPUDispatcher::~CPUDispatcher( void )
{
  // nothing to do here
}

void aasd::CPUDispatcher::DispatchJob( aass::ExecutionRequest& jobRequest )
{
  // simply run the job
  adal::Solver& solver = jobRequest.GetSolver();
  solver.ConnectListener(*this);
  solver.Run(jobRequest.GetTimeline(), jobRequest.GetNumericalModel());
  solver.DisconnectListener(*this);
}

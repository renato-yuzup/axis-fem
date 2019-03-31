#include "AnalysisStep_Pimpl.hpp"

namespace aaj = axis::application::jobs;
namespace adal = axis::domain::algorithms;
namespace ada = axis::domain::analyses;
namespace aao = axis::application::output;

aaj::AnalysisStep::Pimpl::Pimpl( adal::Solver& s, ada::AnalysisTimeline& tl, aao::ResultBucket& bucket ) :
  solver(s), timeline(tl), resultBucket(bucket)
{
  // nothing to do here
}

aaj::AnalysisStep::Pimpl::~Pimpl( void )
{
  solver.Destroy();
  timeline.Destroy();
  resultBucket.Destroy();
  accelerations.DestroyChildren();
  velocities.DestroyChildren();
  displacements.DestroyChildren();
  locks.DestroyChildren();
  nodalLoads.DestroyChildren();
}

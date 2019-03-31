#include "ExecutionRequest.hpp"
#include "ExecutionRequest.hpp"

namespace ada  = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace aass = axis::application::scheduling::scheduler;

aass::ExecutionRequest::ExecutionRequest( adal::Solver& solver, 
  ada::AnalysisTimeline& timeline, ada::NumericalModel& model, 
  const axis::String& jobName ) : solver_(solver), timeline_(timeline), 
  model_(model), jobName_(jobName)
{
  // nothing to do here
}

aass::ExecutionRequest::~ExecutionRequest( void )
{
  // nothing to do here
}

adal::Solver& aass::ExecutionRequest::GetSolver( void )
{
  return solver_;
}

ada::AnalysisTimeline& aass::ExecutionRequest::GetTimeline( void )
{
  return timeline_;
}

ada::NumericalModel& aass::ExecutionRequest::GetNumericalModel( void )
{
  return model_;
}

axis::String aass::ExecutionRequest::GetJobName( void ) const
{
  return jobName_;
}

#include "AnalysisStepInformation.hpp"

namespace aaj = axis::application::jobs;
namespace asdi = axis::services::diagnostics::information;
namespace ada = axis::domain::analyses;
namespace afd = axis::foundation::date_time;
namespace afu = axis::foundation::uuids;

aaj::AnalysisStepInformation::AnalysisStepInformation( const axis::String& stepName, 
                                                       int stepIndex, 
                                                       const asdi::SolverCapabilities& solverCaps, 
                                                       const ada::AnalysisTimeline& timeline, 
                                                       WorkFolder& jobWorkFolder, 
                                                       const axis::String& jobTitle, 
                                                       const afu::Uuid& jobId, 
                                                       const afd::Timestamp& startTime ) :
stepName_(stepName), stepIndex_(stepIndex), solverCaps_(solverCaps), timeline_(timeline),
jobWorkFolder_(jobWorkFolder), jobTitle_(jobTitle), jobId_(jobId), startTime_(startTime)
{
  // nothing to do here
}

axis::String aaj::AnalysisStepInformation::GetStepName( void ) const
{
  return stepName_;
}

int aaj::AnalysisStepInformation::GetStepIndex( void ) const
{
  return stepIndex_;
}

asdi::SolverCapabilities aaj::AnalysisStepInformation::GetSolverCapabilities( void ) const
{
  return solverCaps_;
}

const ada::AnalysisTimeline& aaj::AnalysisStepInformation::GetStepTimeline( void ) const
{
  return timeline_;
}

aaj::WorkFolder& aaj::AnalysisStepInformation::GetJobWorkFolder( void ) const
{
  return jobWorkFolder_;
}

axis::String aaj::AnalysisStepInformation::GetJobTitle( void ) const
{
  return jobTitle_;
}

afu::Uuid aaj::AnalysisStepInformation::GetJobId( void ) const
{
  return jobId_;
}

afd::Timestamp aaj::AnalysisStepInformation::GetJobStartTime( void ) const
{
  return startTime_;
}

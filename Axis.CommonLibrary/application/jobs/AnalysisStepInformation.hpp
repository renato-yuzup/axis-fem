#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"
#include "services/diagnostics/information/SolverCapabilities.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace axis { 

namespace domain { namespace analyses {
class AnalysisTimeline;
}} // namespace axis::domain::analyses

namespace application { namespace jobs {

class WorkFolder;
  
class AXISCOMMONLIBRARY_API AnalysisStepInformation
{
public:
  AnalysisStepInformation(const axis::String& stepName,
                          int stepIndex,
                          const axis::services::diagnostics::information::SolverCapabilities& solverCaps,
                          const axis::domain::analyses::AnalysisTimeline& timeline,
                          WorkFolder& jobWorkFolder,
                          const axis::String& jobTitle,
                          const axis::foundation::uuids::Uuid& jobId,
                          const axis::foundation::date_time::Timestamp& startTime);

  axis::String GetStepName(void) const;
  int GetStepIndex(void) const;
  axis::services::diagnostics::information::SolverCapabilities GetSolverCapabilities(void) const;
  const axis::domain::analyses::AnalysisTimeline& GetStepTimeline(void) const;
  WorkFolder& GetJobWorkFolder(void) const;
  axis::String GetJobTitle(void) const;
  axis::foundation::uuids::Uuid GetJobId(void) const;
  axis::foundation::date_time::Timestamp GetJobStartTime(void) const;
private:
  axis::String stepName_;
  int stepIndex_;
  axis::services::diagnostics::information::SolverCapabilities solverCaps_;
  const axis::domain::analyses::AnalysisTimeline& timeline_;
  WorkFolder& jobWorkFolder_;
  axis::String jobTitle_;
  axis::foundation::uuids::Uuid jobId_;
  axis::foundation::date_time::Timestamp startTime_;
};

} } } // namespace axis::application::jobs

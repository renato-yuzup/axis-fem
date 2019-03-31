#pragma once
#include "StructuralAnalysis.hpp"
#include <vector>
#include "monitoring/HealthMonitor.hpp"

namespace axis { namespace application { namespace jobs {

class StructuralAnalysis::Pimpl
{
public:
  typedef std::vector<AnalysisStep *> step_list;

  Pimpl(void);
  ~Pimpl(void);

  monitoring::HealthMonitor monitor;
  step_list steps;
  axis::domain::analyses::NumericalModel *model;
  WorkFolder *workFolder;
  
  axis::String title;
  axis::foundation::date_time::Timestamp creationDate;
  axis::foundation::uuids::Uuid jobId;
  axis::foundation::date_time::Timestamp analysisStartTime;
  axis::foundation::date_time::Timestamp analysisEndTime;
  int currentStepIndex;
};

} } } // namespace axis::application::jobs

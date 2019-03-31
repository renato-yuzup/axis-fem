#pragma once
#include "AxisString.hpp"
#include "services/messaging/CollectorHub.hpp"

namespace axis {

namespace foundation { 
namespace uuids {
class Uuid;
} // namespace axis::foundation::uuids
namespace date_time {
class Timestamp;
} // namespace axis::foundation::date_time
} // namespace axis::foundation

namespace domain {

namespace algorithms {
class Solver;
} // namespace algorithms

namespace analyses {
  class AnalysisTimeline;
  class NumericalModel;
} // namespace analysis

} // namespace domain

namespace application {

namespace output {
class ResultBucket;
} // namespace output

namespace post_processing {
class PostProcessor;
} // namespace post_processing

namespace jobs {

namespace monitoring {
class HealthMonitor;
} // namespace monitoring

class AnalysisStep;
class WorkFolder;

class FiniteElementAnalysis : public axis::services::messaging::CollectorHub
{
public:
  FiniteElementAnalysis(axis::domain::analyses::NumericalModel& model,
                        axis::domain::algorithms::Solver& solver,
                        axis::domain::analyses::AnalysisTimeline& timeline,
                        axis::application::output::ResultBucket& resultArchives,
                        axis::application::jobs::WorkFolder& outputFolder);
  ~FiniteElementAnalysis(void);

  void AttachMonitor(axis::application::jobs::monitoring::HealthMonitor& monitor);
  void AttachPostProcessor(axis::application::post_processing::PostProcessor& postProcessor);
  void SetJobInformation(const axis::String& jobName, const axis::foundation::uuids::Uuid& jobId,
                         const axis::foundation::date_time::Timestamp& jobStartTime);
  void SetStepInformation(const axis::String& stepName, int stepIndex);

  void StartAnalysis(void);
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} // namespace jobs
} // namespace application
} // namespace axis

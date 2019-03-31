#pragma once
#include "FiniteElementAnalysis.hpp"
#include "services/messaging/MessageListener.hpp"
#include "application/output/ResultBucket.hpp"
#include "application/jobs/monitoring/HealthMonitor.hpp"
#include "application/post_processing/PostProcessor.hpp"
#include "foundation/uuids/Uuid.hpp"

namespace aajm = axis::application::jobs::monitoring;
namespace aao = axis::application::output;
namespace aapp = axis::application::post_processing;
namespace ada = axis::domain::analyses;

namespace {

class SolverForwarderMessageListener : public axis::services::messaging::MessageListener
{
public:
  SolverForwarderMessageListener(const ada::NumericalModel& model, aajm::HealthMonitor *monitor,
                                 aapp::PostProcessor *postProcessor, aao::ResultBucket& resultArchives) :
      model_(model), monitor_(monitor), postProcessor_(postProcessor), 
      resultArchives_(resultArchives) { }

  ~SolverForwarderMessageListener(void) { }
private:
  virtual void DoProcessResultMessage(axis::services::messaging::ResultMessage& volatileMessage)
  {
    // forward messages in the correct order
    if (postProcessor_ != NULL) postProcessor_->ProcessResult(volatileMessage);
    resultArchives_.PlaceResult(volatileMessage, model_);
    if (monitor_ != NULL) monitor_->ProcessMessage(volatileMessage);
  }

  const axis::domain::analyses::NumericalModel& model_;
  axis::application::jobs::monitoring::HealthMonitor *monitor_;
  axis::application::post_processing::PostProcessor *postProcessor_;
  axis::application::output::ResultBucket &resultArchives_;
};

} // namespace

namespace axis { namespace application { namespace jobs {

class FiniteElementAnalysis::Pimpl
{
public:
  Pimpl(void);

  axis::String jobName;
  axis::foundation::uuids::Uuid jobId;
  axis::foundation::date_time::Timestamp jobStartTime;
  
  axis::String stepName;
  int stepIndex;

  axis::domain::analyses::NumericalModel             *model;
  axis::domain::algorithms::Solver                   *solver;
  axis::domain::analyses::AnalysisTimeline           *timeline;
  axis::application::output::ResultBucket            *resultArchives;
  axis::application::jobs::monitoring::HealthMonitor *monitor;
  axis::application::post_processing::PostProcessor  *postProcessor;
  axis::application::jobs::WorkFolder                *workFolder;
  SolverForwarderMessageListener                     *solverListener;
};

} } } // namespace axis::application::jobs

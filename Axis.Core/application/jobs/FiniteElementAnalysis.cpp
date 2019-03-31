#include "FiniteElementAnalysis.hpp"
#include "FiniteElementAnalysis_Pimpl.hpp"
#include "application/output/ResultBucket.hpp"
#include "application/jobs/monitoring/HealthMonitor.hpp"
#include "domain/algorithms/Solver.hpp"
#include "application/jobs/AnalysisStepInformation.hpp"
#include "application/output/collectors/messages/AnalysisStepStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisStepEndMessage.hpp"
#include "application/scheduling/scheduler/ExecutionRequest.hpp"
#include "application/scheduling/scheduler/ExecutionScheduler.hpp"

namespace aaj = axis::application::jobs;
namespace aajm = axis::application::jobs::monitoring;
namespace aaocm = axis::application::output::collectors::messages;
namespace ada = axis::domain::analyses;
namespace adal = axis::domain::algorithms;
namespace aao = axis::application::output;
namespace aass = axis::application::scheduling::scheduler;
namespace afd = axis::foundation::date_time;
namespace afu = axis::foundation::uuids;

aaj::FiniteElementAnalysis::FiniteElementAnalysis(ada::NumericalModel& model,
                                                  adal::Solver& solver,
                                                  ada::AnalysisTimeline& timeline,
                                                  aao::ResultBucket& resultArchives,
                                                  aaj::WorkFolder& outputFolder)
{
  pimpl_                  = new Pimpl();
  pimpl_->model           = &model;
  pimpl_->solver          = &solver;
  pimpl_->timeline        = &timeline;
  pimpl_->resultArchives  = &resultArchives;
  pimpl_->workFolder      = &outputFolder;
}

aaj::FiniteElementAnalysis::~FiniteElementAnalysis(void)
{
  delete pimpl_;
}

void aaj::FiniteElementAnalysis::AttachMonitor( aajm::HealthMonitor& monitor )
{
  pimpl_->monitor= &monitor;
}

void aaj::FiniteElementAnalysis::AttachPostProcessor( aapp::PostProcessor& postProcessor )
{
  pimpl_->postProcessor = &postProcessor;
}

void aaj::FiniteElementAnalysis::SetJobInformation( const axis::String& jobName, 
                                                    const afu::Uuid& jobId, 
                                                    const afd::Timestamp& jobStartTime )
{
  pimpl_->jobName = jobName;
  pimpl_->jobId = jobId;
  pimpl_->jobStartTime = jobStartTime;
}

void aaj::FiniteElementAnalysis::SetStepInformation( const axis::String& stepName, int stepIndex )
{
  pimpl_->stepName = stepName;
  pimpl_->stepIndex = stepIndex;
}

void aaj::FiniteElementAnalysis::StartAnalysis(void)
{
  ada::NumericalModel& model = *pimpl_->model;
  ada::AnalysisTimeline& timeline = *pimpl_->timeline;
  adal::Solver& solver = *pimpl_->solver;
  aao::ResultBucket& resultArchives = *pimpl_->resultArchives;
  aaj::WorkFolder& workFolder = *pimpl_->workFolder;
  aajm::HealthMonitor *monitor = pimpl_->monitor;
  aapp::PostProcessor *postProcessor = pimpl_->postProcessor;  
  SolverForwarderMessageListener listener(model, monitor, postProcessor, resultArchives);
  aaj::AnalysisStepInformation info(pimpl_->stepName, pimpl_->stepIndex, 
                                    solver.GetCapabilities(), timeline, workFolder, 
                                    pimpl_->jobName, pimpl_->jobId, pimpl_->jobStartTime);
  listener.ProcessMessage(aaocm::AnalysisStepStartMessage(info));
  model.InitStep();
  ProcessMessage(aaocm::AnalysisStepStartMessage(info));
  
  // try to schedule analysis for processing
  aass::ExecutionRequest request(solver, timeline, model, pimpl_->stepName);
  aass::ExecutionScheduler& scheduler = aass::ExecutionScheduler::GetActive();
  scheduler.ConnectListener(listener);
  scheduler.ConnectListener(*this);
  bool submitted = scheduler.Submit(request);
  scheduler.DisconnectListener(listener);
  scheduler.DisconnectListener(*this);
  if (!submitted)
  {
    // TODO: Take actions if analysis has not been submitted!
  }
  ProcessMessage(aaocm::AnalysisStepEndMessage());
  listener.ProcessMessage(aaocm::AnalysisStepEndMessage());
}

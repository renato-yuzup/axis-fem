#pragma once
#include "domain/fwd/numerical_model.hpp"
#include "domain/fwd/solver_fwd.hpp"
#include "QueueStatistics.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace broker {

/**
 * Encapsulates algorithms which decides which resource type is better 
 * suited for a job.
**/
class SchedulerAi
{
public:
  SchedulerAi(void);
  ~SchedulerAi(void);

  /**
   * Analyzes a job and return if it is interesting to run it on GPU.
   *
   * @param model  The job numerical model.
   * @param solver The job solver.
   *
   * @return true if it is better to run on GPU, false otherwise.
  **/
  bool ShouldRunOnGPU(const axis::domain::analyses::NumericalModel& model,
                      const axis::domain::algorithms::Solver& solver);

  /**
   * Evaluates the processing queue state and returns if it is affordable to 
   * queue a job in the GPU queue.
   *
   * @param statistics The processing queue statistics.
   * @param model      The job numerical model.
   * @param solver     The job solver.
   *
   * @return true if it is affordable, false otherwise.
  **/
  bool EvaluateGPUQueue(const QueueStatistics& statistics,
                        const axis::domain::analyses::NumericalModel& model,
                        const axis::domain::algorithms::Solver& solver);
};

} } } } // namespace axis::application::scheduling::broker

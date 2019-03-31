#pragma once
#include "foundation/Axis.Capsicum.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "domain/fwd/solver_fwd.hpp"
#include "AxisString.hpp"

namespace axis { namespace application { namespace scheduling { namespace scheduler {

/**
 * Denotes a processing request.
 */
class AXISCAPSICUM_API ExecutionRequest
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] solver   The solver to be run.
   * @param [in,out] timeline The analysis timeline.
   * @param [in,out] model    The numerical model.
   * @param jobName           Job name.
   */
  ExecutionRequest(axis::domain::algorithms::Solver& solver,
                   axis::domain::analyses::AnalysisTimeline& timeline,
                   axis::domain::analyses::NumericalModel& model,
                   const axis::String& jobName);
  ~ExecutionRequest(void);

  /**
   * Returns the solver to be run.
   *
   * @return The solver.
   */
  axis::domain::algorithms::Solver& GetSolver(void);

  /**
   * Returns the analysis timeline.
   *
   * @return The timeline.
   */
  axis::domain::analyses::AnalysisTimeline& GetTimeline(void);

  /**
   * Returns the numerical model.
   *
   * @return The numerical model.
   */
  axis::domain::analyses::NumericalModel& GetNumericalModel(void);

  /**
   * Returns the job name.
   *
   * @return The job name.
   */
  axis::String GetJobName(void) const;
private:
  axis::domain::algorithms::Solver& solver_;
  axis::domain::analyses::AnalysisTimeline& timeline_;
  axis::domain::analyses::NumericalModel& model_;
  axis::String jobName_;
};

} } } } // namespace axis::application::scheduling::scheduler

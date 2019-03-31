#pragma once
#include "domain/fwd/numerical_model.hpp"
#include "domain/fwd/solver_fwd.hpp"
#include "SchedulerType.hpp"
#include "nocopy.hpp"

namespace axis { namespace application { namespace scheduling { namespace scheduler {

/**
 * Checks if a job request is valid for dispatching.
**/
class JobInspector
{
public:
  JobInspector(void);
  ~JobInspector(void);

  /**
   * Inspects numerical model and solver in order to verify that all entities
   * have at least one supported processing hardware.
   *
   * @param model  The numerical model.
   * @param solver The solver.
   *
   * @return true if it succeeds, false if it fails.
   */
  bool operator()(const axis::domain::analyses::NumericalModel& model, 
                  const axis::domain::algorithms::Solver& solver);

  /**
   * Inspects numerical model and solver in order to verify that all entities
   * have at least one supported processing hardware.
   *
   * @param model  The numerical model.
   * @param solver The solver.
   *
   * @return true if it succeeds, false if it fails.
   */
  bool Inspect(const axis::domain::analyses::NumericalModel& model, 
               const axis::domain::algorithms::Solver& solver);

  /**
   * Returns one of the common processing hardware for the numerical model and
   * solver or, if specified, processing hardware requested by the numerical
   * model/solver.
   *
   * @return The job requested resource.
   */
  SchedulerType GetJobRequestedResource(void) const;
private:
  SchedulerType resourceType_;

  DISALLOW_COPY_AND_ASSIGN(JobInspector);
};

} } } } // namespace axis::application::scheduling::scheduler

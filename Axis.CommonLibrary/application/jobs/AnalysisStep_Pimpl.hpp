#include "AnalysisStep.hpp"
#include "domain/collections/BoundaryConditionCollection.hpp"
#include "application/output/ResultBucket.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"
#include "domain/algorithms/Solver.hpp"
#include "application/post_processing/PostProcessor.hpp"

namespace axis { namespace application { namespace jobs {

class AnalysisStep::Pimpl
{
public:
  Pimpl(axis::domain::algorithms::Solver& solver, 
        axis::domain::analyses::AnalysisTimeline& timeline,
        axis::application::output::ResultBucket& bucket);
  ~Pimpl(void);

  axis::domain::algorithms::Solver &solver;
  axis::application::post_processing::PostProcessor postProcessor;
  axis::domain::analyses::AnalysisTimeline &timeline;
  axis::domain::collections::BoundaryConditionCollection accelerations;
  axis::domain::collections::BoundaryConditionCollection velocities;
  axis::domain::collections::BoundaryConditionCollection displacements;
  axis::domain::collections::BoundaryConditionCollection locks;
  axis::domain::collections::BoundaryConditionCollection nodalLoads;
  axis::application::output::ResultBucket& resultBucket;
  axis::String name;
  real startTime;
  real endTime;
};

} } } // namespace axis::application::jobs

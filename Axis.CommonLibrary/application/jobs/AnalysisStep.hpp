#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/algorithms/Solver.hpp"
#include "services/messaging/CollectorHub.hpp"
#include "domain/collections/BoundaryConditionCollection.hpp"
#include "foundation/collections/ObjectList.hpp"

namespace axis {
namespace application {

namespace output {
class ResultBucket;
} // namespace output

namespace post_processing {
class PostProcessor;
} // namespace post_processing

namespace jobs {
/**********************************************************************************************//**
	* <summary> Defines an analysis step composed of a phenomenon timeline, an associated
	* 			 solver for this time frame and any collectors for the generated results.</summary>
	*
	* <seealso cref="axis::services::messaging::CollectorHub"/>
	**************************************************************************************************/
class AXISCOMMONLIBRARY_API AnalysisStep
{
public:
	~AnalysisStep(void);

  /**
   * Creates a new step.
   *
   * @param startTime       The start time of the step time frame.
   * @param endTime         The end time of the step time frame.
   * @param [in,out] solver The associated solver for this step.
   * @param [in,out] bucket The object that will manage results for the step.
   *
   * @return A new analysis step object.
   */
	static AnalysisStep& Create(real startTime, real endTime, axis::domain::algorithms::Solver& solver,
                              axis::application::output::ResultBucket& bucket);
	void Destroy(void) const;

	const axis::domain::algorithms::Solver& GetSolver(void) const;
	axis::domain::algorithms::Solver& GetSolver(void);
  const axis::application::post_processing::PostProcessor& GetPostProcessor(void) const;
  axis::application::post_processing::PostProcessor& GetPostProcessor(void);

	const axis::domain::analyses::AnalysisTimeline& GetTimeline(void) const;
	axis::domain::analyses::AnalysisTimeline& GetTimeline(void);
	real GetStartTime(void) const;
	real GetEndTime(void) const;
  const axis::application::output::ResultBucket& GetResults(void) const;
  axis::application::output::ResultBucket& GetResults(void);

  const axis::domain::collections::BoundaryConditionCollection& NodalLoads(void) const;
	axis::domain::collections::BoundaryConditionCollection& NodalLoads(void);
	const axis::domain::collections::BoundaryConditionCollection& Accelerations(void) const;
	axis::domain::collections::BoundaryConditionCollection& Accelerations(void);
	const axis::domain::collections::BoundaryConditionCollection& Velocities(void) const;
	axis::domain::collections::BoundaryConditionCollection& Velocities(void);
	const axis::domain::collections::BoundaryConditionCollection& Displacements(void) const;
	axis::domain::collections::BoundaryConditionCollection& Displacements(void);
	const axis::domain::collections::BoundaryConditionCollection& Locks(void) const;
	axis::domain::collections::BoundaryConditionCollection& Locks(void);

  bool DefinesBoundaryCondition(axis::domain::elements::DoF& dof) const;
	axis::String GetName(void) const;
	void SetName(const axis::String& name);
private:
  class Pimpl;
  Pimpl *pimpl_;

  AnalysisStep(const AnalysisStep& other);
  AnalysisStep& operator = (const AnalysisStep& other);
  AnalysisStep(real startTime, real endTime, axis::domain::algorithms::Solver& solver,
               axis::application::output::ResultBucket& bucket);
};

} // namespace jobs
} // namespace application
} // namespace axis


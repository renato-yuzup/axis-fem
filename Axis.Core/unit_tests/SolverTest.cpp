#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"
#include "System.hpp"
#include "AxisString.hpp"
#include "MockSolver.hpp"
#include "MockResultBucket.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/output/ResultBucket.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/analyses/SnapshotMark.hpp"
#include "MockClockwork.hpp"

namespace aao = axis::application::output;

using namespace axis::foundation;
using namespace axis::application::jobs;
using namespace axis::domain::analyses;
using namespace axis::domain::algorithms;

namespace axis { namespace unit_tests { namespace core {

TEST_CLASS(SolverTest)
{
private:
	axis::application::jobs::StructuralAnalysis& CreateTestBenchWorkspace(void) const
	{
		StructuralAnalysis& analysis = *new StructuralAnalysis(_T("."));
		analysis.SetNumericalModel(NumericalModel::Create());
		return analysis;
	}
public:
  TEST_METHOD_INITIALIZE(SetUp)
  {
    axis::System::Initialize();
  }

  TEST_METHOD_CLEANUP(TearDown)
  {
    axis::System::Finalize();
  }

	TEST_METHOD(TestStepWalkingWithRegularInterval)
	{
		// create test environment
		StructuralAnalysis& analysis = CreateTestBenchWorkspace();
		MockSolver& solver = *new MockSolver(*new MockClockwork(0.25));
		AnalysisStep& step = AnalysisStep::Create(0, 1.5, solver, *new MockResultBucket());
		analysis.AddStep(step);
		step.GetTimeline().AddSnapshotMark(SnapshotMark(0));
		step.GetTimeline().AddSnapshotMark(SnapshotMark(0.75));
		step.GetTimeline().AddSnapshotMark(SnapshotMark(1.5));

		solver.Run(step.GetTimeline(), analysis.GetNumericalModel());

		Assert::AreEqual(true, solver.IsStepNestingOk());
		Assert::AreEqual(6, solver.GetSecondaryStepCount());
		Assert::AreEqual(2, solver.GetPrimaryStepCount());
	}

	TEST_METHOD(TestStepWalkingWithIrregularInterval)
	{
		// create test environment
		StructuralAnalysis& analysis = CreateTestBenchWorkspace();
		MockSolver& solver = *new MockSolver(*new MockClockwork(0.25));
		AnalysisStep& step = AnalysisStep::Create(0, 2.0, solver, *new MockResultBucket());
		analysis.AddStep(step);
		step.GetTimeline().AddSnapshotMark(SnapshotMark(0));
		step.GetTimeline().AddSnapshotMark(SnapshotMark(0.5));
		step.GetTimeline().AddSnapshotMark(SnapshotMark(2.0));

		solver.Run(step.GetTimeline(), analysis.GetNumericalModel());

		Assert::AreEqual(true, solver.IsStepNestingOk());
		Assert::AreEqual(8, solver.GetSecondaryStepCount());
		Assert::AreEqual(2, solver.GetPrimaryStepCount());
	}
};

} } }

#endif

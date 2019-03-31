#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "TestConjugateGradientSolver.hpp"
#include "foundation/blas/blas.hpp"
#include "domain/algorithms/RegularClockwork.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"

using namespace axis::foundation::blas;
using namespace axis::domain::algorithms;
using namespace axis::domain::analyses;
using namespace axis::foundation;

namespace axis { namespace unit_tests { namespace AxisSolver {

/* ================================================================================================================== */
/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
TEST_CLASS(ConjugateGradientTestFixture)
{
public:
  TEST_METHOD_INITIALIZE(SetUp)
  {
    axis::System::Initialize();
  }

  TEST_METHOD_CLEANUP(TearDown)
  {
    axis::System::Finalize();
  }

	TEST_METHOD(TestSolve)
	{
		RegularClockwork& clock = *new RegularClockwork(1);
		TestConjugateGradientSolver solver(clock);
		NumericalModel& analysis = NumericalModel::Create();

		// create timeline
		AnalysisTimeline& timeline = AnalysisTimeline::Create(0, 1);
		timeline.AddSnapshotMark(SnapshotMark(0));

		solver.ToggleDebug(true);
		solver.ToggleMute(false);
		solver.Run(timeline, analysis);

		const ColumnVector& x = solver.GetSolutionVector();

		real x0 = x(0);
		real x1 = x(1);
		real x2 = x(2);
		real x3 = x(3);
		real x4 = x(4);
		real x5 = x(5);
		real x6 = x(6);
		real x7 = x(7);

		Assert::AreEqual(3,  x0, REAL_TOLERANCE);
		Assert::AreEqual(5,  x1, REAL_TOLERANCE);
		Assert::AreEqual(-1, x2, REAL_TOLERANCE);
		Assert::AreEqual(8,  x3, REAL_TOLERANCE);
		Assert::AreEqual(10, x4, REAL_TOLERANCE);
		Assert::AreEqual(7,  x5, REAL_TOLERANCE);
		Assert::AreEqual(9,  x6, REAL_TOLERANCE);
		Assert::AreEqual(0,  x7, REAL_TOLERANCE);

		delete &analysis;
	}

};

} } }

#endif

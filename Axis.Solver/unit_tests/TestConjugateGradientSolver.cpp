#if defined DEBUG || defined _DEBUG

#include "TestConjugateGradientSolver.hpp"
#include "foundation/blas/blas.hpp"
#include "foundation/memory/pointer.hpp"

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace asdi = axis::services::diagnostics::information;

TestConjugateGradientSolver::TestConjugateGradientSolver(axis::domain::algorithms::Clockwork& clock) :
ConjugateGradientSolver(clock)
{
	// nothing to do
}

TestConjugateGradientSolver::~TestConjugateGradientSolver(void)
{
	// nothing to do
}


void TestConjugateGradientSolver::ExecutePostProcessing( const axis::foundation::blas::ColumnVector& solutionVector, 
                                                         axis::domain::analyses::NumericalModel& analysis, 
                                                         long iterationCount, 
                                                         const axis::domain::analyses::AnalysisTimeline& timeInfo )
{
	// nothing to do
}

void TestConjugateGradientSolver::ExecuteInitialSteps( axis::domain::analyses::NumericalModel& analysis )
{

#if defined(AXIS_NO_MEMORY_ARENA)
	x0 = new axis::foundation::blas::ColumnVector(8);
	A  = new axis::foundation::blas::DenseMatrix(8, 8);
	Q  = new axis::foundation::blas::ColumnVector(8);
	R0 = new axis::foundation::blas::ColumnVector(8);
#define S(x,y, val)	A->SetElement(x, y, val);	A->SetElement(y, x, val);
#define B(x, val)	(*R0)(x) = val
#else
  x0 = axis::foundation::blas::ColumnVector::CreateFromGlobalMemory(8);
  A  = axis::foundation::blas::DenseMatrix::CreateFromGlobalMemory(8, 8);
  Q  = axis::foundation::blas::ColumnVector::CreateFromGlobalMemory(8);
  R0 = axis::foundation::blas::ColumnVector::CreateFromGlobalMemory(8);
#define S(x,y, val)	axis::absref<afb::DenseMatrix>(A)(x, y) = val;	axis::absref<afb::DenseMatrix>(A)(y, x) = val;
#define B(x, val)	axis::absref<afb::ColumnVector>(R0)(x) = val
#endif

	S(0, 0, 8);
	S(1, 0, -1)	S(1, 1, 5);
	S(2, 0, 6)	S(2, 1, 2)	S(2, 2, 3);
	S(3, 0, 7)	S(3, 1, 4)	S(3, 2, 0)	S(3, 3, 12);
	S(4, 0, 5)	S(4, 1, 3)	S(4, 2, 8)	S(4, 3, 9)	S(4, 4, 1);
	S(5, 0, 1)	S(5, 1, 1)	S(5, 2, -5)	S(5, 3, 2)	S(5, 4, 5)	S(5, 5, 9);
	S(6, 0, 10)	S(6, 1, 7)	S(6, 2, 3)	S(6, 3, 6)	S(6, 4, 8)	S(6, 5, 1)	S(6, 6, 2);
	S(7, 0, 8)	S(7, 1, -3)	S(7, 2, 6)	S(7, 3, -9)	S(7, 4, 5)	S(7, 5, 1)	S(7, 6, 10)	S(7, 7, 6);

	B(0, 216); B(1, 152); B(2, 97); B(3, 295); B(4, 211); B(5, 151); B(6, 215); B(7, 78);

	axis::absref<afb::ColumnVector>(x0).ClearAll();
}

void TestConjugateGradientSolver::ExecuteCleanupSteps( axis::domain::analyses::NumericalModel& analysis )
{
#if defined(AXIS_NO_MEMORY_ARENA)
	delete A;
	delete Q;
	delete R0;
#else
  axis::absref<afb::DenseMatrix>(A).Destroy();
  axis::absref<afb::ColumnVector>(Q).Destroy();
  axis::absref<afb::ColumnVector>(R0).Destroy();
  axis::System::GlobalMemory().Deallocate(A);
  axis::System::GlobalMemory().Deallocate(Q);
  axis::System::GlobalMemory().Deallocate(R0);
#endif
}
void TestConjugateGradientSolver::WarnLongWait( long currentIterationStep )
{
	// nothing to do here
}

void TestConjugateGradientSolver::AbortSolutionProcedure( long lastIterationStep )
{
	// nothing to do
}

afb::ColumnVector& TestConjugateGradientSolver::GetInitializedResidualWorkVector( void )
{
	return axis::absref<afb::ColumnVector>(R0);
}

afb::ColumnVector& TestConjugateGradientSolver::GetInitializedSolutionWorkVector( void )
{
  return axis::absref<afb::ColumnVector>(x0);
}

afb::ColumnVector& TestConjugateGradientSolver::AssembleQ( const axis::foundation::blas::ColumnVector& searchDirectionVector )
{
	afb::VectorProduct(axis::absref<afb::ColumnVector>(Q), 1.0, axis::absref<afb::DenseMatrix>(A), searchDirectionVector);
	return axis::absref<afb::ColumnVector>(Q);
}

long TestConjugateGradientSolver::GetMaximumIterationsAllowed( void ) const
{
	return 100;
}

long TestConjugateGradientSolver::GetNumStepsToLongComputation( void ) const
{
	return 10;
}

int TestConjugateGradientSolver::GetSolverEventSourceId( void ) const
{
	return 0;
}

real TestConjugateGradientSolver::CalculateRhsVectorNorm( void )
{
#if defined(AXIS_NO_MEMORY_ARENA)
  return R0->Norm();
#else
  return axis::absref<afb::ColumnVector>(R0).Norm();
#endif
}

real TestConjugateGradientSolver::CalculateRhsScalarProduct( const afb::ColumnVector& rightFactor )
{
#if defined(AXIS_NO_MEMORY_ARENA)
	return afb::VectorAlgebra::ScalarProduct(*R0, rightFactor);
#else
  return afb::VectorScalarProduct(axis::absref<afb::ColumnVector>(R0), rightFactor);
#endif
}

void TestConjugateGradientSolver::Destroy( void ) const
{
	delete this;
}

asdi::SolverCapabilities TestConjugateGradientSolver::GetCapabilities( void ) const
{
	return asdi::SolverCapabilities(
							_T("TestConjugateGradientSolver"),
							_T(""), 
							false, false, false, false);
}

#endif

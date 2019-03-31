#pragma once
#include "domain/algorithms/ConjugateGradientSolver.hpp"
#include "foundation/blas/blas.hpp"

class TestConjugateGradientSolver : public axis::domain::algorithms::ConjugateGradientSolver
{
protected:
#if defined(AXIS_NO_MEMORY_ARENA)
	axis::foundation::blas::Matrix *A;
	axis::foundation::blas::ColumnVector *x0;
	axis::foundation::blas::ColumnVector *R0;
	axis::foundation::blas::ColumnVector *Q;
#else
  axis::foundation::memory::RelativePointer A;
  axis::foundation::memory::RelativePointer x0;
  axis::foundation::memory::RelativePointer R0;
  axis::foundation::memory::RelativePointer Q;
#endif

	virtual void ExecuteInitialSteps( axis::domain::analyses::NumericalModel& analysis );

	virtual void ExecutePostProcessing( const axis::foundation::blas::ColumnVector& solutionVector, 
                                      axis::domain::analyses::NumericalModel& analysis, 
                                      long iterationCount, 
                                      const axis::domain::analyses::AnalysisTimeline& timeInfo );

	virtual void ExecuteCleanupSteps( axis::domain::analyses::NumericalModel& analysis );

	virtual void WarnLongWait( long currentIterationStep );
	virtual void AbortSolutionProcedure( long lastIterationStep );

	virtual real CalculateRhsScalarProduct( const axis::foundation::blas::ColumnVector& rightFactor );
	virtual real CalculateRhsVectorNorm( void );

	virtual axis::foundation::blas::ColumnVector& GetInitializedResidualWorkVector( void );
	virtual axis::foundation::blas::ColumnVector& GetInitializedSolutionWorkVector( void );
	virtual axis::foundation::blas::ColumnVector& AssembleQ( const axis::foundation::blas::ColumnVector& searchDirectionVector );

	virtual long GetMaximumIterationsAllowed( void ) const;
	virtual long GetNumStepsToLongComputation( void ) const;
	
	virtual int GetSolverEventSourceId(void) const;
public:
	TestConjugateGradientSolver(axis::domain::algorithms::Clockwork& clock);
	~TestConjugateGradientSolver(void);

	virtual void Destroy( void ) const;

	virtual axis::services::diagnostics::information::SolverCapabilities GetCapabilities( void ) const;
};


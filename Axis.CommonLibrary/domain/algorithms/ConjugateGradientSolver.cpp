#include "ConjugateGradientSolver.hpp"
#include <limits>

#include "foundation/InvalidOperationException.hpp"
#if !defined(AXIS_NO_MEMORY_ARENA)
#include "foundation/memory/pointer.hpp"
#endif

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

#ifdef AXIS_DOUBLEPRECISION
	const real axis::domain::algorithms::ConjugateGradientSolver::ErrorTolerance = 1e-12;
#else
	const real axis::domain::algorithms::ConjugateGradientSolver::ErrorTolerance = 1e-6;
#endif

const int axis::domain::algorithms::ConjugateGradientSolver::MaxConvergenceDelay = 1000;
const real axis::domain::algorithms::ConjugateGradientSolver::ConvergenceHeuristicsThreshold = 1.01;
const int axis::domain::algorithms::ConjugateGradientSolver::ConvergenceHeuristicsDelayIncrement = 20;


axis::domain::algorithms::ConjugateGradientSolver::ConjugateGradientSolver( Clockwork& clockwork ) : 
Solver(clockwork)
{
	_finalSolutionVector = NULLPTR;
	_phiVector = new real[MaxConvergenceDelay];

	_solutionVector = NULL;
	_newSolutionVector = NULLPTR;
	_searchDirection = NULLPTR;
	_initialResidualVector = NULLPTR;
	_residualVector = NULL;
}

axis::domain::algorithms::ConjugateGradientSolver::~ConjugateGradientSolver( void )
{
	delete [] _phiVector;

	// destroy other objects if we haven't done so yet
	if (_finalSolutionVector != NULLPTR)
  {
#if defined(AXIS_NO_MEMORY_ARENA)
    _finalSolutionVector->Destroy();
#else
    ((afb::ColumnVector *)*_finalSolutionVector)->Destroy();
    System::GlobalMemory().Deallocate(_finalSolutionVector);
#endif
  }
	if (_initialResidualVector != NULLPTR)
  {
#if defined(AXIS_NO_MEMORY_ARENA)
    _initialResidualVector->Destroy();
#else
    ((afb::ColumnVector *)*_initialResidualVector)->Destroy();
    System::GlobalMemory().Deallocate(_initialResidualVector);
#endif
  }
	_finalSolutionVector = NULLPTR;
	_initialResidualVector = NULLPTR;
}

bool axis::domain::algorithms::ConjugateGradientSolver::Converged( const axis::foundation::blas::ColumnVector& currentSolutionVector, const axis::foundation::blas::ColumnVector& initialResidualVector, real rho )
{
	// force iteration if we haven't reached a suitable delay
	if (_lastCsi == std::numeric_limits<real>::infinity()) return false;
	
	real rhs = rho + afb::VectorScalarProduct(currentSolutionVector, initialResidualVector);
	rhs *= _muTolerance;
	return _lastCsi <= rhs;
}

void axis::domain::algorithms::ConjugateGradientSolver::CalculateConvergenceHeuristics( real phi )
{
	// save new phi value to phi history vector
	_phiVector[_phiVectorWritePos++] = phi;

	// start overwriting positions if no more room is available
	if (_phiVectorWritePos >= MaxConvergenceDelay)
	{
		_phiVectorWritePos = 0;
		_overflowFlag = true;
	}

	// calculate csi
	real csi = CalculateCsi(_delay);

	// compare with last csi value
	if (_lastCsi != std::numeric_limits<real>::infinity())
	{
		if (csi * ConvergenceHeuristicsThreshold > _lastCsi)
		{
			_delay = _delay + ConvergenceHeuristicsDelayIncrement;
			_delay = _delay > MaxConvergenceDelay? MaxConvergenceDelay : _delay;

			// recalculate csi
			csi = CalculateCsi(_delay);
		}
	}

	_lastCsi = csi;
}

real axis::domain::algorithms::ConjugateGradientSolver::CalculateCsi( int delay )
{
	// check if we have enough history to calculate csi
	bool ok = (_phiVectorWritePos >= delay) || _overflowFlag;

	if (!ok)
	{	// insufficient data; just return the last result
		return _lastCsi;
	}

	int count = 0;
	int i = _phiVectorWritePos == 0? MaxConvergenceDelay - 1 : _phiVectorWritePos - 1;
	real csi = 0;
	while (count < _delay)
	{
		csi += _phiVector[i];
		i = (i == 0)? MaxConvergenceDelay - 1 : i - 1;
		++count;
	}

	return csi;
}

void axis::domain::algorithms::ConjugateGradientSolver::EnterSecondaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model )
{

	// delete old solution
	if (_finalSolutionVector != NULLPTR)
	{
#if defined(AXIS_NO_MEMORY_ARENA)
		delete _finalSolutionVector;
#else
    ((afb::ColumnVector *)*_finalSolutionVector)->Destroy();
    System::GlobalMemory().Deallocate(_finalSolutionVector);
#endif
	}
	_finalSolutionVector = NULLPTR;

	/***************************************************************
	* STEP 1 -- DO INITIAL STEPS
	* --------------------------------------------------------------
	* Do all the necessary tasks required to run the algorithm.
	***************************************************************/
	LogSolverMessage(_T("I'm doing the preliminary steps prior to the iterative solution..."));
	ExecuteInitialSteps(model);

	/***************************************************************
	* STEP 2 -- CALCULATE INITIAL VECTORS
	* --------------------------------------------------------------
	* The first guess of the solution vector, the residual vector
	* and search direction vector needs to be calculated.
	***************************************************************/
	// we need the initial residual and solution vectors
	LogSolverMessage(_T("Calculating initial vectors..."));

	// we suppose these tasks can run independently
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			_solutionVector = &GetInitializedSolutionWorkVector();	// vector {P}		
		}
		#pragma omp section
		{
      _residualVector = &GetInitializedResidualWorkVector();		// vector {R0}
			_rhsVectorNorm = CalculateRhsVectorNorm();			// norm of {b}
#if defined(AXIS_NO_MEMORY_ARENA)
			_rho = CalculateRhsScalarProduct(*_solutionVector);	// {b}^T * {U0}
#else
      _rho = CalculateRhsScalarProduct(*_solutionVector);	// {b}^T * {U0}
#endif
		}
	}

	// create a copy of the initial residual vector -- we will need
	// it to calculate algorithm convergence
  afb::ColumnVector& residualVector = *_residualVector;
  afb::ColumnVector& solutionVector = *_solutionVector;
  _initialResidualVector = afb::ColumnVector::CreateFromGlobalMemory(residualVector.Length());
#if defined(AXIS_NO_MEMORY_ARENA)
  _initialResidualVector->CopyFrom(*_residualVector);
#else
  ((afb::ColumnVector *)*_initialResidualVector)->CopyFrom(residualVector);
#endif

	// fail if residual and solution vectors do not match in size
	if (solutionVector.Rows() != residualVector.Rows() || 
      solutionVector.Columns() != residualVector.Columns() || residualVector.Columns() != 1)
	{
		LogSolverMessage(_T("Invalid solver behavior detected. R0 and x0 sizes are invalid or do not match."));
		throw axis::foundation::InvalidOperationException();
	}

	// initialize search direction vector
#if defined(AXIS_NO_MEMORY_ARENA)
	_searchDirection = new afb::ColumnVector(_residualVector->Length());
	_searchDirection->CopyFrom(*_residualVector);
#else
  _searchDirection = afb::ColumnVector::CreateFromGlobalMemory(residualVector.Length());
  ((afb::ColumnVector *)*_searchDirection)->CopyFrom(residualVector);
#endif

	// allocate solution and residual vectors (for a n+1 step)
#if defined(AXIS_NO_MEMORY_ARENA)
	_newSolutionVector = new afb::ColumnVector(_residualVector->Length());
#else
  _newSolutionVector = afb::ColumnVector::CreateFromGlobalMemory(residualVector.Length());
#endif

	// DEBUG: print vectors
	LogDebugMessage(_T("---- BEFORE ITERATION ----------------------------------------"));
  LogDebugVector(*_solutionVector, _T("X_0"));
  LogDebugVector(*_residualVector, _T("R_0"));
	LogDebugVector(*(afb::ColumnVector *)*_searchDirection, _T("P_1"));
}

void axis::domain::algorithms::ConjugateGradientSolver::ExecuteStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model )
{
	/*
		The algorithm implemented here is based on the following reference:

		GOLUB, G.H. & VAN LOAN, C.F. (1996) Matrix Computations, 3rd ed., John Hopkins, pp. 520-527

		Convergence criterion used is adapted from:

		ARIOLI, M. A stopping criterion for the Conjugate Gradient Algorithm in a Finite Element 
		Method Framework. Tech Report no. RAL-TR-2002-034, Rutherford Appleton Laboratory, 2002.
	*/

	// CG main loop
	LogSolverMessage(_T("Going to start iterative method..."));

	_iterationCount = 0;

	_lastCsi = std::numeric_limits<real>::infinity();
	_delay = 10;	// our initial delay
	_phiVectorWritePos = 0;
	_overflowFlag = false;
	_muTolerance = ErrorTolerance;

	// store pointer to the vector which will contain the final
	// solution
	afb::ColumnVector *finalSolution = absptr<afb::ColumnVector>(_newSolutionVector);
  afb::ColumnVector *newSolutionPtr = absptr<afb::ColumnVector>(_newSolutionVector);

  while (!Converged(*_solutionVector, absref<afb::ColumnVector>(_initialResidualVector), _rho))
	{
    afb::ColumnVector& residualVector = *_residualVector;
    afb::ColumnVector& newSolutionVector = *newSolutionPtr;
    afb::ColumnVector& solutionVector = *_solutionVector;
    afb::ColumnVector& searchDirection = absref<afb::ColumnVector>(_searchDirection);

    ++_iterationCount;
		if (_iterationCount > GetMaximumIterationsAllowed())
		{	// it took too much time; finish now
			AbortSolutionProcedure(--_iterationCount);
			break;
		}
		if (_iterationCount % GetNumStepsToLongComputation() == 0)
		{	// warn that CG might take a long time to finish
			WarnLongWait(_iterationCount);
		}

		LogDebugMessage(_T("---- ITERATION ") + String::int_parse(_iterationCount) + _T(" ----------------------------------------"));

		// calculate [A] * {P};
    afb::ColumnVector& Q = AssembleQ(searchDirection);
		if (IsDebugEnabled())
		{
			LogDebugVector(searchDirection, _T("P_") + String::int_parse(_iterationCount-1));
			LogDebugVector(Q, _T("Q_") + String::int_parse(_iterationCount-1));
		}

		real alphaNumerator, alphaDenominator;
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				alphaNumerator = residualVector.SelfScalarProduct();
			}
			#pragma omp section
			{
				alphaDenominator = afb::VectorScalarProduct(searchDirection, Q);
			}
		}
		real alpha = alphaNumerator / alphaDenominator;
		LogDebugScalar(alpha, _T("alpha"));

		// these operations cal also be calculated in parallel
		real beta;
		real phi;
		if (IsDebugEnabled())
		{
			LogDebugVector(solutionVector, _T("X_") + String::int_parse(_iterationCount - 1));
		}
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				// calculate new solution vector
				afb::VectorSum(newSolutionVector, solutionVector, alpha, searchDirection);
			}
			#pragma omp section
			{
				// calculate beta and update residual vector
				real betaDenominator = residualVector.SelfScalarProduct();	// scalar product of old residual
				phi = alpha * betaDenominator;
				afb::VectorSum(residualVector, residualVector, -alpha, Q);
				real betaNumerator = residualVector.SelfScalarProduct();		// scalar product of updated residual
				beta  = betaNumerator / betaDenominator;
			}
		}
		if (IsDebugEnabled())
		{
			LogDebugVector(newSolutionVector, _T("X_") + String::int_parse(_iterationCount));
			LogDebugVector(residualVector, _T("R_") + String::int_parse(_iterationCount));
			LogDebugScalar(beta, _T("beta"));
		}

		// update search direction
		afb::VectorSum(searchDirection, residualVector, beta, searchDirection);
		if (IsDebugEnabled())
		{
			LogDebugVector(searchDirection, _T("P_") + String::int_parse(_iterationCount+1));
		}

		// swap pointers (as if x(n-1) = x(n))
		afb::ColumnVector *aux = _solutionVector;
		_solutionVector = newSolutionPtr;
		newSolutionPtr = aux;

		// calculate heuristics
		CalculateConvergenceHeuristics(phi);
	}
	if (IsDebugEnabled())
	{
		LogDebugMessage(_T("-- Finished after ") + String::int_parse(_iterationCount) + _T(" iterations."));
	}

	// yes, this code is right; because we swapped pointers in the last loop,
	// the most accurate solution is in the solutionVector pointer
	if (_solutionVector != finalSolution)
	{
		// we don't have the solution in our vector; copy it		
		finalSolution->CopyFrom(*_solutionVector);
	}
	_finalSolutionVector = _newSolutionVector;
}

void axis::domain::algorithms::ConjugateGradientSolver::ExitSecondaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model )
{
#if defined(AXIS_NO_MEMORY_ARENA)
	ExecutePostProcessing(*_finalSolutionVector, model, _iterationCount, timeline);
#else
  ExecutePostProcessing(*(afb::ColumnVector *)*_finalSolutionVector, model, _iterationCount, timeline);
#endif
}

void axis::domain::algorithms::ConjugateGradientSolver::ExitPrimaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model )
{
	// do cleanup
#if defined(AXIS_NO_MEMORY_ARENA)
	_initialResidualVector->Destroy();
#else
  ((afb::ColumnVector *)*_initialResidualVector)->Destroy();
  System::GlobalMemory().Deallocate(_initialResidualVector);
#endif
	_initialResidualVector = NULLPTR;
	ExecuteCleanupSteps(model);
}

const axis::foundation::blas::ColumnVector& axis::domain::algorithms::ConjugateGradientSolver::GetSolutionVector( void ) const
{
	if (_finalSolutionVector == NULLPTR)
	{
		throw axis::foundation::InvalidOperationException(_T("Solution vector unavailable."));
	}
	return *(afb::ColumnVector *)*_finalSolutionVector;
}

const axis::domain::analyses::AnalysisInfo& axis::domain::algorithms::ConjugateGradientSolver::GetAnalysisInformation( void ) const
{
  return analysisInfo_;
}

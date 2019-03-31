#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "Solver.hpp"
#include "domain/analyses/StaticAnalysisInfo.hpp"
#if !defined(AXIS_NO_MEMORY_ARENA)
#include "foundation/memory/RelativePointer.hpp"
#endif

namespace axis
{
	namespace domain
	{
		namespace algorithms
		{
			/**********************************************************************************************//**
			 * <summary> Implements a linear static solver that solves by using the Conjugate 
			 * 			 Gradient Method.</summary>
			 *
			 * <seealso cref="Solver"/>
			 **************************************************************************************************/
			class AXISCOMMONLIBRARY_API ConjugateGradientSolver : public Solver
			{
			public:

				/**********************************************************************************************//**
				 * <summary> The error tolerance.</summary>
				 **************************************************************************************************/
				static const real ErrorTolerance;

				/**********************************************************************************************//**
				 * <summary> The maximum number of iterations stipulated to reach convergence.</summary>
				 **************************************************************************************************/
				static const int MaxConvergenceDelay;

				/**********************************************************************************************//**
				 * <summary> The tolerable increase ratio of csi before increasing the delay value.</summary>
				 **************************************************************************************************/
				static const real ConvergenceHeuristicsThreshold;

				/**********************************************************************************************//**
				 * <summary> Amount by which delay is incremented.</summary>
				 **************************************************************************************************/
				static const int ConvergenceHeuristicsDelayIncrement;

				/**********************************************************************************************//**
				 * <summary> Constructor.</summary>
				 *
				 * <param name="clockwork"> [in,out] Associated clockwork to be used by this solver to advance 
				 * 							analysis time.</param>
				 **************************************************************************************************/
				ConjugateGradientSolver(Clockwork& clockwork);

				/**********************************************************************************************//**
				 * <summary> Destructor.</summary>
				 **************************************************************************************************/
				virtual ~ConjugateGradientSolver(void);

				/**********************************************************************************************//**
				 * <summary> Returns, if exist, the solution vector.</summary>
				 *
				 * <returns> The solution vector.</returns>
				 * <remarks> If no solution vector exists (or have not yet been found), an exception occurs.
				 * 			 </remarks>
				 **************************************************************************************************/
				const axis::foundation::blas::ColumnVector& GetSolutionVector(void) const;

        virtual const axis::domain::analyses::AnalysisInfo& GetAnalysisInformation( void ) const;
      private:
				/**********************************************************************************************//**
				 * <summary> When overridden, executes any action needed before entering the next secondary step.
				 * 			 </summary>
				 *
				 * <param name="timeline"> The analysis timeline.</param>
				 * <param name="model">    [in,out] The numerical model in analysis.</param>
				 **************************************************************************************************/
				virtual void EnterSecondaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model );

				/**********************************************************************************************//**
				 * <summary> When overridden, executes the computations pertaining to the current secondary step.
				 * 			 </summary>
				 *
				 * <param name="timeline"> The analysis timeline.</param>
				 * <param name="model">    [in,out] The numerical model in analysis.</param>
				 **************************************************************************************************/
				virtual void ExecuteStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model );

				/**********************************************************************************************//**
				 * <summary> When overridden, executes actions needed before leaving the current secondary step.
				 * 			 </summary>
				 *
				 * <param name="timeline"> The analysis timeline.</param>
				 * <param name="model">    [in,out] The numerical model in analysis.</param>
				 **************************************************************************************************/
				virtual void ExitSecondaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model );

				/**********************************************************************************************//**
				 * <summary> When overridden, executes actions needed before leaving the current primary step.</summary>
				 *
				 * <param name="timeline"> The analysis timeline.</param>
				 * <param name="model">    [in,out] The numerical model in analysis.</param>
				 **************************************************************************************************/
				virtual void ExitPrimaryStep( const axis::domain::analyses::AnalysisTimeline& timeline, axis::domain::analyses::NumericalModel& model );


        axis::domain::analyses::StaticAnalysisInfo analysisInfo_;
			protected:
#if defined(AXIS_NO_MEMORY_ARENA)
				axis::foundation::blas::Vector *_finalSolutionVector;
#else
        axis::foundation::memory::RelativePointer _finalSolutionVector;
#endif
				real _rhsVectorNorm;

				// these are needed to calculate convergence of the algorithm
				real *_phiVector;
				int _phiVectorWritePos;
				int _delay;
				bool _overflowFlag;
				real _lastCsi;
				real _muTolerance;


				// internal variables for the conjugate gradient method
#if defined(AXIS_NO_MEMORY_ARENA)
        axis::foundation::blas::ColumnVector *_solutionVector;		    // last solution vector
        axis::foundation::blas::ColumnVector *_newSolutionVector;		  // solution vector obtained in current step
        axis::foundation::blas::ColumnVector *_searchDirection;		    // solution search direction vector (gradient)
        axis::foundation::blas::ColumnVector *_initialResidualVector;	// first residual vector
        axis::foundation::blas::ColumnVector *_residualVector;		    // current step residual vector
#else
				axis::foundation::blas::ColumnVector *_solutionVector;		        // last solution vector
				axis::foundation::memory::RelativePointer _newSolutionVector;		  // solution vector obtained in current step
				axis::foundation::memory::RelativePointer _searchDirection;		    // solution search direction vector (gradient)
				axis::foundation::memory::RelativePointer _initialResidualVector;	// first residual vector
        axis::foundation::blas::ColumnVector *_residualVector;		        // current step residual vector
#endif
				long _iterationCount;									// number of convergence iterations
				real _rho;												// a scalar used in the process

				/**********************************************************************************************//**
				 * <summary> Executes any initialization step for the algorithm.</summary>
				 *
				 * <param name="analysis"> [in,out] The numerical model being analyzed.</param>
				 **************************************************************************************************/
				virtual void ExecuteInitialSteps(axis::domain::analyses::NumericalModel& analysis) = 0;

				/**********************************************************************************************//**
				 * <summary> Executes the post processing operation after solution
				 * 			 convergence.</summary>
				 *
				 * <param name="solutionVector"> The solution vector.</param>
				 * <param name="analysis">		 [in,out] The numerical model analyzed.</param>
				 * <param name="iterationCount"> Number of iterations needed to
				 * 								 converge.</param>
				 **************************************************************************************************/
				virtual void ExecutePostProcessing(const axis::foundation::blas::ColumnVector& solutionVector, axis::domain::analyses::NumericalModel& analysis, long iterationCount, const axis::domain::analyses::AnalysisTimeline& timeInfo) = 0;

				/**********************************************************************************************//**
				 * <summary> Executes cleanup steps after analysis and post-processing operations.</summary>
				 *
				 * <param name="analysis"> [in,out] The numerical model that was analyzed.</param>
				 **************************************************************************************************/
				virtual void ExecuteCleanupSteps(axis::domain::analyses::NumericalModel& analysis) = 0;

				/**********************************************************************************************//**
				 * <summary> Executes any necessary actions when the algorithm main loop seems to
				 * 			 be taking too much iterations to converge.</summary>
				 *
				 * <param name="currentIterationStep"> The current iteration step number.</param>
				 **************************************************************************************************/
				virtual void WarnLongWait(long currentIterationStep) = 0;

				/**********************************************************************************************//**
				 * <summary> Executes any necessary actions when the converge loop is
				 * 			 aborted on request.</summary>
				 *
				 * <param name="lastIterationStep"> The last iteration step number.</param>
				 **************************************************************************************************/
				virtual void AbortSolutionProcedure(long lastIterationStep) = 0;

				/**********************************************************************************************//**
				 * <summary> Calculates the norm of the right-hand side vector of the linear system being solved.
				 * 			 </summary>
				 *
				 * <returns> A rea number representing the vector norm.</returns>
				 **************************************************************************************************/
				virtual real CalculateRhsVectorNorm(void) = 0;

				/**********************************************************************************************//**
				 * <summary> Calculates the scalar product of the right-hand side vector of the linear system
				 * 			 being solved (say, R) with a given vector (say, X), that is, the result of R^T*X.
				 * 			 </summary>
				 *
				 * <param name="rightFactor"> The right vector (X).</param>
				 *
				 * <returns> The calculated scalar product.</returns>
				 **************************************************************************************************/
				virtual real CalculateRhsScalarProduct(const axis::foundation::blas::ColumnVector& rightFactor) = 0;

				/**********************************************************************************************//**
				 * <summary> Returns an already initialized work vector which will store the residuals.</summary>
				 *
				 * <returns> A reference to the initial residual vector.</returns>
				 **************************************************************************************************/
				virtual axis::foundation::blas::ColumnVector& GetInitializedResidualWorkVector( void ) = 0;

				/**********************************************************************************************//**
				 * <summary> Returns an already initialized work vector .</summary>
				 *
				 * <returns> A reference to the initial solution vector.</returns>
				 **************************************************************************************************/
				virtual axis::foundation::blas::ColumnVector& GetInitializedSolutionWorkVector( void ) = 0;

				/**********************************************************************************************//**
				 * <summary> Returns a reference to the vector which contains the product of the
				 * 			 coefficient matrix by the search direction vector.</summary>
				 *
				 * <param name="searchDirectionVector"> The search direction vector.</param>
				 *
				 * <returns> A reference to a vector.</returns>
				 **************************************************************************************************/
				virtual axis::foundation::blas::ColumnVector& AssembleQ(const axis::foundation::blas::ColumnVector& searchDirectionVector) = 0;

        /**********************************************************************************************//**
				 * <summary> Returns the maximum number of iterations allowed for the convergence loop.</summary>
				 *
				 * <returns> The maximum iterations allowed.</returns>
				 **************************************************************************************************/
				virtual long GetMaximumIterationsAllowed(void) const = 0;

				/**********************************************************************************************//**
				 * <summary> Returns the number steps which defines that the algorithm main loop is
				 * 			 taking too long to converge.</summary>
				 *
				 * <returns> The number of steps.</returns>
				 **************************************************************************************************/
				virtual long GetNumStepsToLongComputation(void) const = 0;

				/**********************************************************************************************//**
				 * <summary> Returns if convergence has been achieved.</summary>
				 *
				 * <param name="currentSolutionVector"> The current solution vector.</param>
				 * <param name="initialResidualVector"> The initial residual vector.</param>
				 * <param name="rho">				    The scalar product of the RHS vector by the 
				 * 										initial solution vector.</param>
				 *
				 * <returns> true if it converged, false otherwise.</returns>
				 **************************************************************************************************/
				virtual bool Converged(const axis::foundation::blas::ColumnVector& currentSolutionVector, const axis::foundation::blas::ColumnVector& initialResidualVector, real rho);

				/**********************************************************************************************//**
				 * <summary> Computes heuristics that defines the convergence ot the method.</summary>
				 *
				 * <param name="phi"> The phi value ((r^T*r)^2/(p^T*q)).</param>
				 **************************************************************************************************/
				virtual void CalculateConvergenceHeuristics(real phi);

				/**********************************************************************************************//**
				 * <summary> Calculates the convergence heuristics scalar csi.</summary>
				 *
				 * <param name="delay"> The minimum number of iterations stipulated to reach convergence.</param>
				 *
				 * <returns> The calculated csi value.</returns>
				 **************************************************************************************************/
				virtual real CalculateCsi(int delay);

      };
		}
	}
}


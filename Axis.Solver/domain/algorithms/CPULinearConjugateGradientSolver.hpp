#pragma once
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "domain/algorithms/ConjugateGradientSolver.hpp"
#include "foundation/date_time/Timestamp.hpp"
#if !defined(AXIS_NO_MEMORY_ARENA)
#include "foundation/memory/RelativePointer.hpp"
#endif

namespace axis { namespace domain { namespace algorithms {

/**********************************************************************************************//**
	* <summary> Implements a linear static solver using the Conjugate Gradient
	* 			 Method running on CPU.</summary>
	*
	* <seealso cref="ConjugateGradientSolver"/>
	**************************************************************************************************/
class CPULinearConjugateGradientSolver : public ConjugateGradientSolver
{
private:
	axis::domain::analyses::NumericalModel *_analysis;

	axis::foundation::date_time::Timestamp _algorithmStartTime;

	// tells if we already have calculated the stiffness of 
	// each element
	bool _elementaryStiffnessCalculated;

	// our work vectors
#if defined(AXIS_NO_MEMORY_ARENA)
  axis::foundation::blas::ColumnVector *_Qn;
  axis::foundation::blas::ColumnVector *_R0;
  axis::foundation::blas::ColumnVector *_U0;
#else
  axis::foundation::memory::RelativePointer _Qn;
  axis::foundation::blas::ColumnVector *_R0;
  axis::foundation::blas::ColumnVector *_U0;
#endif
protected:
	/**********************************************************************************************//**
		* <summary> Executes any initialization step for the algorithm.</summary>
		*
		* <param name="analysis"> [in,out] The numerical model being analyzed.</param>
		**************************************************************************************************/
	virtual void ExecuteInitialSteps( axis::domain::analyses::NumericalModel& analysis );

	/**********************************************************************************************//**
		* <summary> Executes the post processing operation after solution
		* 			 convergence.</summary>
		*
		* <param name="solutionVector"> The solution vector.</param>
		* <param name="analysis">		 [in,out] The numerical model analyzed.</param>
		* <param name="iterationCount"> Number of iterations needed to
		* 								 converge.</param>
		**************************************************************************************************/
	virtual void ExecutePostProcessing( const axis::foundation::blas::ColumnVector& solutionVector, 
                                      axis::domain::analyses::NumericalModel& analysis, 
                                      long iterationCount, 
                                      const axis::domain::analyses::AnalysisTimeline& timeInfo );

	/**********************************************************************************************//**
		* <summary> Executes cleanup steps after analysis and post-processing operations.</summary>
		*
		* <param name="analysis"> [in,out] The numerical model that was analyzed.</param>
		**************************************************************************************************/
	virtual void ExecuteCleanupSteps( axis::domain::analyses::NumericalModel& analysis );

	/**********************************************************************************************//**
		* <summary> Executes any necessary actions when the algorithm main loop seems to
		* 			 be taking too much iterations to converge.</summary>
		*
		* <param name="currentIterationStep"> The current iteration step number.</param>
		**************************************************************************************************/
	virtual void WarnLongWait( long currentIterationStep );

	/**********************************************************************************************//**
		* <summary> Executes any necessary actions when the converge loop is
		* 			 aborted on request.</summary>
		*
		* <param name="lastIterationStep"> The last iteration step number.</param>
		**************************************************************************************************/
	virtual void AbortSolutionProcedure( long lastIterationStep );

	/**********************************************************************************************//**
		* <summary> Calculates the norm of the right-hand side vector of the linear system being solved.
		* 			 </summary>
		*
		* <returns> A rea number representing the vector norm.</returns>
		**************************************************************************************************/
	virtual real CalculateRhsVectorNorm( void );

	/**********************************************************************************************//**
		* <summary> Calculates the scalar product of the right-hand side vector of the linear system
		* 			 being solved (say, R) with a given vector (say, X), that is, the result of R^T*X.
		* 			 </summary>
		*
		* <param name="rightFactor"> The right vector (X).</param>
		*
		* <returns> The calculated scalar product.</returns>
		**************************************************************************************************/
	virtual real CalculateRhsScalarProduct( const axis::foundation::blas::ColumnVector& rightFactor );

  /**********************************************************************************************//**
		* <summary> Returns an already initialized work vector which will store the residuals.</summary>
		*
		* <returns> A reference to the initial residual vector.</returns>
		**************************************************************************************************/
	virtual axis::foundation::blas::ColumnVector& GetInitializedResidualWorkVector( void );

	/**********************************************************************************************//**
		* <summary> Returns an already initialized work vector .</summary>
		*
		* <returns> A reference to the initial solution vector.</returns>
		**************************************************************************************************/
	virtual axis::foundation::blas::ColumnVector& GetInitializedSolutionWorkVector( void );

	/**********************************************************************************************//**
		* <summary> Returns a reference to the vector which contains the product of the
		* 			 coefficient matrix by the search direction vector.</summary>
		*
		* <param name="searchDirectionVector"> The search direction vector.</param>
		*
		* <returns> A reference to a vector.</returns>
		**************************************************************************************************/
	virtual axis::foundation::blas::ColumnVector& AssembleQ( const axis::foundation::blas::ColumnVector& searchDirectionVector );

  /**********************************************************************************************//**
		* <summary> Returns the maximum number of iterations allowed for the convergence loop.</summary>
		*
		* <returns> The maximum iterations allowed.</returns>
		**************************************************************************************************/
	virtual long GetMaximumIterationsAllowed( void ) const;

	/**********************************************************************************************//**
		* <summary> Returns the number steps which defines that the algorithm main loop is
		* 			 taking too long to converge.</summary>
		*
		* <returns> The number of steps.</returns>
		**************************************************************************************************/
	virtual long GetNumStepsToLongComputation( void ) const;

private:

	/**********************************************************************************************//**
		* <summary> Assembles the global external load vector.</summary>
		*
		* <param name="externalLoadVector"> [in,out] The vector where data should be written.</param>
		**************************************************************************************************/
	void AssembleExternalLoadVector( axis::foundation::blas::ColumnVector& externalLoadVector );

	/**********************************************************************************************//**
		* <summary> Applies prescribed displacements to load vector
		* 			 described by externalLoadVector.</summary>
		*
		* <param name="externalLoadVector"> [in,out] The external load vector.</param>
		**************************************************************************************************/
	void ApplyPrescribedDisplacementsToLoadVector( axis::foundation::blas::ColumnVector& externalLoadVector );

	/**********************************************************************************************//**
		* <summary> Calculates a coefficient of the Q vector.</summary>
		*
		* <param name="node">	   The node correspondent to the coefficient.</param>
		* <param name="P">		   The search direction vector.</param>
		* <param name="dofIndex"> Zero-based index of the degree of freedom.</param>
		*
		* <returns> The calculated coefficient.</returns>
		**************************************************************************************************/
	real CalculateQCoefficient( const axis::domain::elements::Node& node, const axis::foundation::blas::ColumnVector& P, int dofIndex ) const;

	/**********************************************************************************************//**
		* <summary> Queries if a degree of freedom of a node is 
		* 			 constrained.</summary>
		*
		* <param name="node">	   The node.</param>
		* <param name="dofIndex"> Zero-based index of the degree of freedom.</param>
		*
		* <returns> true if degree of freedom is constrained, false otherwise.</returns>
		**************************************************************************************************/
	bool IsDofMovementConstrained(const axis::domain::elements::Node& node, int dofIndex) const;

	/**********************************************************************************************//**
		* <summary> Calculates the stiffness matrix of all elements in the model.</summary>
		**************************************************************************************************/
	void CalculateElementsStiffnessMatrix( void );

	/**********************************************************************************************//**
		* <summary> Starts a section header in the log.</summary>
		*
		* <param name="sectionName"> Name of the section.</param>
		**************************************************************************************************/
	void LogSectionHeader(const axis::String& sectionName) const;

	/**********************************************************************************************//**
		* <summary> Closes a section with a footer in the log.</summary>
		*
		* <param name="sectionName"> Name of the section.</param>
		**************************************************************************************************/
	void LogSectionFooter(const axis::String& sectionName) const;

	/**********************************************************************************************//**
		* <summary> Calculates the elements stresses and strains.</summary>
		*
		* <param name="solutionVector"> The displacement solution vector.</param>
		**************************************************************************************************/
	void CalculateElementsStressStrain(const axis::foundation::blas::ColumnVector& solutionVector, const axis::domain::analyses::AnalysisTimeline& timeInfo);

	/**********************************************************************************************//**
		* <summary> Calculates the nodal stresses and strains.</summary>
		**************************************************************************************************/
	void CalculateNodesStressStrain(void);
public:

	/**********************************************************************************************//**
		* <summary> Constructor.</summary>
		*
		* <param name="clockwork"> [in,out] Associated clockwork to be used by this solver to advance 
		* 							analysis time.</param>
		**************************************************************************************************/
	CPULinearConjugateGradientSolver(Clockwork& clock);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	virtual ~CPULinearConjugateGradientSolver(void);

	/**********************************************************************************************//**
		* <summary> Destroys this object.</summary>
		**************************************************************************************************/
	virtual void Destroy( void ) const;

	/**********************************************************************************************//**
		* <summary> Gets the solver event source identifier.</summary>
		*
		* <returns> The solver event source identifier.</returns>
		**************************************************************************************************/
	virtual int GetSolverEventSourceId(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the capabilities of this solver.</summary>
		*
		* <returns> An object describing the capabilities of this solver.</returns>
		**************************************************************************************************/
	virtual axis::services::diagnostics::information::SolverCapabilities GetCapabilities( void ) const;
};

} } } // namespace axis::domain::algorithms

#pragma once
#include "domain/algorithms/Clockwork.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"

#include "services/messaging/CollectorEndpoint.hpp"
#include "services/diagnostics/information/SolverCapabilities.hpp"

#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/blas/blas.hpp"

namespace axis { namespace domain {
  
namespace analyses {
class AnalysisInfo;
class ReducedNumericalModel;
}

namespace algorithms {

class ExternalSolverFacade;

class AXISCOMMONLIBRARY_API Solver : public axis::services::messaging::CollectorEndpoint
{
public:
	/**********************************************************************************************//**
		* <summary> Constructor.</summary>
		*
		* <param name="clockwork"> [in,out] Associated clockwork to be used by this solver to advance 
		* 							analysis time.</param>
		**************************************************************************************************/
	Solver(Clockwork& clockwork);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	virtual ~Solver(void);

	/**********************************************************************************************//**
		* <summary> Destroys this object.</summary>
		**************************************************************************************************/
	virtual void Destroy(void) const = 0;

	/**********************************************************************************************//**
		* <summary> Queries if debug mode is enabled.</summary>
		*
		* <returns> true if debug is enabled, false otherwise.</returns>
		**************************************************************************************************/
	bool IsDebugEnabled(void) const;

	/**********************************************************************************************//**
		* <summary> Toggles debug mode. When enabled, messages intended for debugging purposes are
		* 			 sent by this object. It is not affected by mute mode.</summary>
		*
		* <param name="state"> true to set debug mode enabled.</param>
		**************************************************************************************************/
	void ToggleDebug(bool state);

	/**********************************************************************************************//**
		* <summary> Toggles verbose mode. In verbose mode, any informational message (except debugging)
		* 			 are sent by this object. It does not have effect if mute mode is enabled.</summary>
		*
		* <param name="state"> true to set verbose mode enabled.</param>
		**************************************************************************************************/
	void ToggleVerbosity(bool state);

	/**********************************************************************************************//**
		* <summary> Toggles mute mode. When mute, only result messages are sent.</summary>
		*
		* <param name="state"> New mute mode state. Setting it to true activates mute mode.</param>
		**************************************************************************************************/
	void ToggleMute(bool state);

	/**********************************************************************************************//**
		* <summary> Gets the solver event source identifier.</summary>
		*
		* <returns> The solver event source identifier.</returns>
		**************************************************************************************************/
	virtual int GetSolverEventSourceId(void) const = 0;

	/**********************************************************************************************//**
		* <summary> Returns the capabilities of this solver.</summary>
		*
		* <returns> An object describing the capabilities of this solver.</returns>
		**************************************************************************************************/
	virtual axis::services::diagnostics::information::SolverCapabilities GetCapabilities(void) const = 0;

	/**********************************************************************************************//**
		* <summary> Starts the analysis process.</summary>
		*
		* <param name="timeline"> [in,out] The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model.</param>
		**************************************************************************************************/
	void Run(axis::domain::analyses::AnalysisTimeline& timeline, 
           axis::domain::analyses::NumericalModel& model);

	/**********************************************************************************************//**
		* <summary> Starts the analysis process on GPU.</summary>
		*
		* <param name="timeline"> [in,out] The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model.</param>
		**************************************************************************************************/
	void RunOnGPU(axis::domain::analyses::AnalysisTimeline& timeline, 
                axis::foundation::memory::RelativePointer& reducedModelPtr,
                axis::domain::algorithms::ExternalSolverFacade& solverFacade);

	/**********************************************************************************************//**
		* <summary> Returns if this solver offers functionality to run in GPU.</summary>
		*
		* <returns> true if this solver is capable, false otherwise.</returns>
		**************************************************************************************************/
  bool IsGPUCapable(void) const;

	/**********************************************************************************************//**
		* <summary> Returns if this solver offers functionality to run in CPU.</summary>
		*
		* <returns> true if this solver is capable, false otherwise.</returns>
		**************************************************************************************************/
  virtual bool IsCPUCapable(void) const;

  /**
   * Allocates memory to solver data and possibly initializes according to numerical 
   * model configuration.
   *
   * @param [in,out] model The numerical model.
  **/
  virtual void AllocateGPUData(axis::domain::analyses::NumericalModel& model,
                               axis::domain::analyses::AnalysisTimeline& timeline);

  /**
   * Returns how many GPU threads are required to run this solver.
   *
   * @param model The numerical model which will be analyzed.
   *
   * @return The GPU threads required.
  **/
  virtual size_type GetGPUThreadsRequired(const axis::domain::analyses::NumericalModel& model) const;

  /**
   * Executes preliminary tasks before memory is transferred to GPU and solving 
   * iterations are started.
  **/
  virtual void PrepareGPUData(axis::domain::analyses::NumericalModel& model, 
                              axis::domain::analyses::AnalysisTimeline& timeline);

  /**
    * Returns current status of the analysis process.
    *
    * @return The analysis information.
    */
  virtual const axis::domain::analyses::AnalysisInfo& GetAnalysisInformation(void) const = 0;
protected:
	/**********************************************************************************************//**
		* <summary> If verbose mode is enabled and mute mode is disabled, logs a solver message.</summary>
		*
		* <param name="message"> The message.</param>
		**************************************************************************************************/
	void LogSolverMessage(const axis::String& message) const;

	/**********************************************************************************************//**
		* <summary> If not in mute mode, logs a solver informational message.</summary>
		*
		* <param name="eventId"> Identifier for the event.</param>
		* <param name="message"> The message.</param>
		**************************************************************************************************/
	void LogSolverInfoMessage(int eventId, const axis::String& message) const;

	/**********************************************************************************************//**
		* <summary> If not in mute mode, logs a solver error message.</summary>
		*
		* <param name="eventId"> Identifier for the event.</param>
		* <param name="message"> The message.</param>
		**************************************************************************************************/
	void LogSolverErrorMessage(int eventId, const axis::String& message) const;

	/**********************************************************************************************//**
		* <summary> It not in mute mode, logs a solver warning message.</summary>
		*
		* <param name="eventId"> Identifier for the event.</param>
		* <param name="message"> The message.</param>
		**************************************************************************************************/
	void LogSolverWarningMessage(int eventId, const axis::String& message) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs the contents of a matrix.</summary>
		*
		* <param name="matrix">	 The matrix.</param>
		* <param name="matrixName"> Name of the matrix.</param>
		**************************************************************************************************/
	void LogDebugMatrix(const axis::foundation::blas::DenseMatrix& matrix, 
                      const axis::String& matrixName) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs the contents of a matrix.</summary>
		*
		* <param name="matrix">	 The matrix.</param>
		* <param name="matrixName"> Name of the matrix.</param>
		**************************************************************************************************/
	void LogDebugMatrix(const axis::foundation::blas::SymmetricMatrix& matrix, 
                      const axis::String& matrixName) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs the contents of a matrix.</summary>
		*
		* <param name="matrix">	 The matrix.</param>
		* <param name="matrixName"> Name of the matrix.</param>
		**************************************************************************************************/
	void LogDebugMatrix(const axis::foundation::blas::LowerTriangularMatrix& matrix, 
                      const axis::String& matrixName) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs the contents of a matrix.</summary>
		*
		* <param name="matrix">	 The matrix.</param>
		* <param name="matrixName"> Name of the matrix.</param>
		**************************************************************************************************/
	void LogDebugMatrix(const axis::foundation::blas::UpperTriangularMatrix& matrix, 
                      const axis::String& matrixName) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs the contents of a vector.</summary>
		*
		* <param name="vector">	  The vector.</param>
		* <param name="vectorName">  Name of the vector.</param>
		* <param name="asRowVector"> (optional) tells if the vector should be treated as a row vector.</param>
		**************************************************************************************************/
	void LogDebugVector(const axis::foundation::blas::ColumnVector& vector, const axis::String& vectorName, 
                      bool asRowVector = false) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs the contents of a vector.</summary>
		*
		* <param name="vector">	  The vector.</param>
		* <param name="vectorName">  Name of the vector.</param>
		* <param name="asColVector"> (optional) tells if the vector should be treated as a column vector.</param>
		**************************************************************************************************/
	void LogDebugVector(const axis::foundation::blas::RowVector& vector, const axis::String& vectorName, 
                      bool asColVector = false) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs a scalar (real number).</summary>
		*
		* <param name="value">		 The scalar value.</param>
		* <param name="scalarName"> Name of the scalar.</param>
		**************************************************************************************************/
	void LogDebugScalar(real value, const axis::String& scalarName) const;

	/**********************************************************************************************//**
		* <summary> If debug mode is enabled, logs a debug message.</summary>
		*
		* <param name="message"> The message.</param>
		**************************************************************************************************/
	void LogDebugMessage(const axis::String& message) const;
private:
	/**********************************************************************************************//**
		* <summary> When overridden, fulfils any necessary prerequisites to start the
		* 			 analysis process.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model in analysis.</param>
		**************************************************************************************************/
	virtual void StartAnalysisProcess(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                    axis::domain::analyses::NumericalModel& model);

	/**********************************************************************************************//**
		* <summary> When overridden, executes any action needed before entering the secondary steps of
		* 			 this primary step.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model in analysis.</param>
		**************************************************************************************************/
	virtual void EnterPrimaryStep(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                axis::domain::analyses::NumericalModel& model);

	/**********************************************************************************************//**
		* <summary> When overridden, executes any action needed before entering the next secondary step.
		* 			 </summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model in analysis.</param>
		**************************************************************************************************/
	virtual void EnterSecondaryStep(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                  axis::domain::analyses::NumericalModel& model);

	/**********************************************************************************************//**
		* <summary> When overridden, executes the computations pertaining to the current secondary step.
		* 			 </summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model in analysis.</param>
		**************************************************************************************************/
	virtual void ExecuteStep(const axis::domain::analyses::AnalysisTimeline& timeline, 
                           axis::domain::analyses::NumericalModel& model);

	/**********************************************************************************************//**
		* <summary> When overridden, executes actions needed before leaving the current secondary step.
		* 			 </summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model in analysis.</param>
		**************************************************************************************************/
	virtual void ExitSecondaryStep(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                 axis::domain::analyses::NumericalModel& model);

	/**********************************************************************************************//**
		* <summary> When overridden, executes actions needed before leaving the current primary step.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model in analysis.</param>
		**************************************************************************************************/
	virtual void ExitPrimaryStep(const axis::domain::analyses::AnalysisTimeline& timeline, 
                               axis::domain::analyses::NumericalModel& model);

	/**********************************************************************************************//**
		* <summary> When overridden, collects final results and ends the analysis process.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] The numerical model in analysis.</param>
		**************************************************************************************************/
	virtual void EndAnalysisProcess(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                  axis::domain::analyses::NumericalModel& model);

  virtual void AddTracingInformation( axis::services::messaging::Message& message ) const;

  /**
   * Returns if this solver is capable to run in GPU.
   *
   * @return  true if it s capable, false otherwise.
  **/
  virtual bool DoIsGPUCapable(void) const;

	/**********************************************************************************************//**
		* <summary> When overridden, fulfils any necessary prerequisites to start the
		* 			    analysis process on GPU.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] Pointer to the numerical model in analysis.</param>
		**************************************************************************************************/
  virtual void StartAnalysisProcessOnGPU(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                         axis::foundation::memory::RelativePointer& reducedModelPtr,
                                         axis::domain::algorithms::ExternalSolverFacade& solverFacade);

	/**********************************************************************************************//**
		* <summary> When overridden, executes any action needed before entering the secondary steps of
		* 			    this primary step on GPU.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] Pointer to the numerical model in analysis.</param>
		**************************************************************************************************/
  virtual void EnterPrimaryStepOnGPU(const axis::domain::analyses::AnalysisTimeline& timeline,  
                                     axis::foundation::memory::RelativePointer& reducedModelPtr,
                                     axis::domain::algorithms::ExternalSolverFacade& solverFacade);
	/**********************************************************************************************//**
		* <summary> When overridden, executes any action needed before entering the next secondary step
		* 			    on GPU.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] Pointer to the numerical model in analysis.</param>
		**************************************************************************************************/
  virtual void EnterSecondaryStepOnGPU(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                       axis::foundation::memory::RelativePointer& reducedModelPtr,
                                       axis::domain::algorithms::ExternalSolverFacade& solverFacade);
	/**********************************************************************************************//**
		* <summary> When overridden, executes the computations pertaining to the current secondary step
		* 			    on GPU.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] Pointer to the numerical model in analysis.</param>
		**************************************************************************************************/
  virtual void ExecuteStepOnGPU(const axis::domain::analyses::AnalysisTimeline& timeline,  
                                axis::foundation::memory::RelativePointer& reducedModelPtr,
                                axis::domain::algorithms::ExternalSolverFacade& solverFacade);
	/**********************************************************************************************//**
		* <summary> When overridden, executes actions needed before leaving the current secondary step
		* 			    on GPU.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] Pointer to the numerical model in analysis.</param>
		**************************************************************************************************/
  virtual void ExitSecondaryStepOnGPU(const axis::domain::analyses::AnalysisTimeline& timeline,  
                                      axis::foundation::memory::RelativePointer& reducedModelPtr,
                                      axis::domain::algorithms::ExternalSolverFacade& solverFacade);
	/**********************************************************************************************//**
		* <summary> When overridden, executes actions needed before leaving the current primary step
    *           on GPU.</summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] Pointer to the numerical model in analysis.</param>
		**************************************************************************************************/
  virtual void ExitPrimaryStepOnGPU(const axis::domain::analyses::AnalysisTimeline& timeline, 
                                    axis::foundation::memory::RelativePointer& reducedModelPtr,
                                    axis::domain::algorithms::ExternalSolverFacade& solverFacade);
	/**********************************************************************************************//**
		* <summary> When overridden, collects final results and ends the analysis process on GPU.
    *           </summary>
		*
		* <param name="timeline"> The analysis timeline.</param>
		* <param name="model">    [in,out] Pointer to the numerical model in analysis.</param>
		**************************************************************************************************/
  virtual void EndAnalysisProcessOnGPU(const axis::domain::analyses::AnalysisTimeline& timeline,  
                                       axis::foundation::memory::RelativePointer& reducedModelPtr,
                                       axis::domain::algorithms::ExternalSolverFacade& solverFacade);

  // definition of solver state
  bool _mute;
  bool _debug;
  bool _verbose;

  // our associated clockwork
  Clockwork& _clock;

  // prevent copy constructor and copy assignment
  Solver(const Solver& other);
  Solver& operator = (const Solver& other);
};

} } }

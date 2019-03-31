#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace axis { namespace domain { namespace analyses {
class ReducedNumericalModel;
} } } // namespace axis::domain::analyses

namespace axis { namespace domain { namespace algorithms {

class ExternalSolverFacade;

/**********************************************************************************************//**
	* <summary> Defines a class responsible for calculating a time increment adequate to 
	* 			 the solver algorithm of an analysis.
	**************************************************************************************************/
class AXISCOMMONLIBRARY_API Clockwork
{
public:

	/**********************************************************************************************//**
		* <summary> Default constructor.</summary>
		**************************************************************************************************/
	Clockwork(void);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	virtual ~Clockwork(void);

	/**********************************************************************************************//**
		* <summary> Calculates and updates the next time increment for the
		* 			 analysis timeline, optionally based on numerical model
		* 			 characteristics.</summary>
		*
		* <param name="ti">    [in,out] The analysis timeline.</param>
		* <param name="model"> [in,out] The numerical model.</param>
		**************************************************************************************************/
	virtual void CalculateNextTick(axis::domain::analyses::AnalysisTimeline& ti, 
                                 axis::domain::analyses::NumericalModel& model) = 0;

	/**********************************************************************************************//**
		* <summary> Calculates and updates the next time increment for the
		* 			 analysis timeline, optionally based on numerical model
		* 			 characteristics.</summary>
		*
		* <param name="ti">			     [in,out] The analysis timeline.</param>
		* <param name="model">			   [in,out] The numerical model.</param>
		* <param name="maxTimeIncrement"> The maximum admissible time increment.</param>
		**************************************************************************************************/
	virtual void CalculateNextTick(axis::domain::analyses::AnalysisTimeline& ti, 
                                 axis::domain::analyses::NumericalModel& model, 
                                 real maxTimeIncrement) = 0;

	/**********************************************************************************************//**
		* <summary> Destroys this object.</summary>
		**************************************************************************************************/
	virtual void Destroy(void) const = 0;

  /**
    * Returns if this object is capable to run in GPU devices.
    *
    * @return true if it is GPU capable, false otherwise.
  **/
  virtual bool IsGPUCapable(void) const;

  virtual void CalculateNextTickOnGPU(axis::domain::analyses::AnalysisTimeline& timeline, 
                                      const axis::foundation::memory::RelativePointer& reducedModelPtr,
                                      axis::domain::algorithms::ExternalSolverFacade& solverFacade);
  virtual void CalculateNextTickOnGPU(axis::domain::analyses::AnalysisTimeline& timeline, 
                                      const axis::foundation::memory::RelativePointer& reducedModelPtr,
                                      axis::domain::algorithms::ExternalSolverFacade& solverFacade,
                                      real maxTimeIncrement);
};

} } } // namespace axis::domain::algorithms

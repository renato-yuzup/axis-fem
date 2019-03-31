#pragma once
#include "domain/algorithms/Clockwork.hpp"

namespace axis
{
	namespace domain
	{
		namespace algorithms
		{
			/**********************************************************************************************//**
			 * <summary> Defines a clockwork which increments time in regular intervals.</summary>
			 *
			 * <seealso cref="Clockwork"/>
			 **************************************************************************************************/
			class RegularClockwork : public Clockwork
			{
			private:
				real _timestepIncrement;
			public:

				/**********************************************************************************************//**
				 * <summary> Default constructor.</summary>
				 *
				 * <param name="timeIncrement"> The time increment.</param>
				 **************************************************************************************************/
				RegularClockwork(real timeIncrement);

				/**********************************************************************************************//**
				 * <summary> Destructor.</summary>
				 **************************************************************************************************/
				~RegularClockwork(void);

				/**************************************************************************************************
				 * <summary>	Creates a new clockwork which works in a fixed time increment basis.</summary>
				 *
				 * <param name="timeIncrement">	The time increment. </param>
				 *
				 * <returns>	A reference to the new object. </returns>
				 **************************************************************************************************/
				static Clockwork& Create(real timeIncrement);

				/**********************************************************************************************//**
				 * <summary> Calculates and updates the next time increment for the
				 * 			 analysis timeline, optionally based on numerical model
				 * 			 characteristics.</summary>
				 *
				 * <param name="ti">    [in,out] The analysis timeline.</param>
				 * <param name="model"> [in,out] The numerical model.</param>
				 **************************************************************************************************/
				virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, axis::domain::analyses::NumericalModel& model );

				/**********************************************************************************************//**
				 * <summary> Calculates and updates the next time increment for the
				 * 			 analysis timeline, optionally based on numerical model
				 * 			 characteristics.</summary>
				 *
				 * <param name="ti">			   [in,out] The analysis timeline.</param>
				 * <param name="model">			   [in,out] The numerical model.</param>
				 * <param name="maxTimeIncrement"> The maximum admissible time increment.</param>
				 **************************************************************************************************/
				virtual void CalculateNextTick( axis::domain::analyses::AnalysisTimeline& ti, axis::domain::analyses::NumericalModel& model, real maxTimeIncrement );

				/**********************************************************************************************//**
				 * <summary> Destroys this object.</summary>
				 **************************************************************************************************/
				virtual void Destroy( void ) const;

        virtual bool IsGPUCapable( void ) const;

        virtual void CalculateNextTickOnGPU(
          axis::domain::analyses::AnalysisTimeline& timeline, 
          const axis::foundation::memory::RelativePointer& reducedModelPtr,
          axis::domain::algorithms::ExternalSolverFacade& solverFacade);

        virtual void CalculateNextTickOnGPU(
          axis::domain::analyses::AnalysisTimeline& timeline, 
          const axis::foundation::memory::RelativePointer& reducedModelPtr,
          axis::domain::algorithms::ExternalSolverFacade& solverFacade,
          real maxTimeIncrement);
      };		
		}
	}
}


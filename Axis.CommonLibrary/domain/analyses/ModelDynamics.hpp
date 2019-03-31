#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis
{
	namespace domain
	{
		namespace analyses
		{
			/**********************************************************************************************//**
			 * <summary> Contains dynamics information about the numerical model.</summary>
			 **************************************************************************************************/
			class AXISCOMMONLIBRARY_API ModelDynamics
			{
			public:
				/**********************************************************************************************//**
				 * <summary> Destructor.</summary>
				 **************************************************************************************************/
				~ModelDynamics(void);

				/**********************************************************************************************//**
				 * <summary> Resets this object.</summary>
				 *
				 * <param name="numDofs"> Number of dofs in the model.</param>
				 **************************************************************************************************/
				void ResetAll(size_type numDofs);

				/**********************************************************************************************//**
				 * <summary> Resets the external loads of the numerical model
				 * 			 setting it all to zero.</summary>
				 *
				 * <param name="numDofs"> Number of dofs in the model.</param>
				 **************************************************************************************************/
				void ResetExternalLoad(size_type numDofs);

				/**********************************************************************************************//**
				 * <summary> Resets the internal forces of the numerical model setting
				 * 			 it all to zero.</summary>
				 *
				 * <param name="numDofs"> Number of dofs in the model.</param>
				 **************************************************************************************************/
				void ResetInternalForce(size_type numDofs);

				/**********************************************************************************************//**
				 * <summary> Resets the effective load of the numerical model setting
				 * 			 it all to zero.</summary>
				 *
				 * <param name="numDofs"> Number of dofs in the model.</param>
				 **************************************************************************************************/
				void ResetReactionForce(size_type numDofs);

				/**********************************************************************************************//**
				 * <summary> Returns the external load vector associated with the numerical model.</summary>
				 *
				 * <returns> The load vector with as many positions as the number of dofs 
				 * 			 in the model.</returns>
				 **************************************************************************************************/
				const axis::foundation::blas::ColumnVector& ExternalLoads(void) const;

				/**********************************************************************************************//**
				 * <summary> Returns the external load vector associated with the numerical model.</summary>
				 *
				 * <returns> The load vector with as many positions as the number of dofs 
				 * 			 in the model.</returns>
				 **************************************************************************************************/
				axis::foundation::blas::ColumnVector& ExternalLoads(void);

				/**********************************************************************************************//**
				 * <summary> Returns the internal force vector associated with the numerical model.</summary>
				 *
				 * <returns> The internal force vector with as many positions as the number of dofs 
				 * 			 in the model.</returns>
				 **************************************************************************************************/
				const axis::foundation::blas::ColumnVector& InternalForces(void) const;

				/**********************************************************************************************//**
				 * <summary> Returns the internal force vector associated with the numerical model.</summary>
				 *
				 * <returns> The internal force vector with as many positions as the number of dofs 
				 * 			 in the model.</returns>
				 **************************************************************************************************/
				axis::foundation::blas::ColumnVector& InternalForces(void);

				/**********************************************************************************************//**
				 * <summary> Returns the effective load vector associated with the numerical model.</summary>
				 *
				 * <returns> The effective load vector with as many positions as the number of dofs 
				 * 			 in the model.</returns>
				 **************************************************************************************************/
				const axis::foundation::blas::ColumnVector& ReactionForce(void) const;

				/**********************************************************************************************//**
				 * <summary> Returns the effective load vector associated with the numerical model.</summary>
				 *
				 * <returns> The effective load vector with as many positions as the number of dofs 
				 * 			 in the model.</returns>
				 **************************************************************************************************/
				axis::foundation::blas::ColumnVector& ReactionForce(void);

				/**********************************************************************************************//**
				 * <summary> Returns a pointer to the the external load vector associated with the 
         *           numerical model.</summary>
				 *
				 * <returns> The relative pointer to the vector.</returns>
				 **************************************************************************************************/
				const axis::foundation::memory::RelativePointer GetExternalLoadsPointer(void) const;

				/**********************************************************************************************//**
				 * <summary> Returns a pointer to the external load vector associated with the 
         *           numerical model.</summary>
				 *
				 * <returns> The relative pointer to the vector.</returns>
				 **************************************************************************************************/
				axis::foundation::memory::RelativePointer GetExternalLoadsPointer(void);

				/**********************************************************************************************//**
				 * <summary> Returns a pointer to the internal force vector associated with the 
         *           numerical model.</summary>
				 *
				 * <returns> The relative pointer to the vector.</returns>
				 **************************************************************************************************/
				const axis::foundation::memory::RelativePointer GetInternalForcesPointer(void) const;

				/**********************************************************************************************//**
				 * <summary> Returns a pointer to the internal force vector associated with the 
         *           numerical model.</summary>
				 *
				 * <returns> The relative pointer to the vector.</returns>
				 **************************************************************************************************/
				axis::foundation::memory::RelativePointer GetInternalForcesPointer(void);

				/**********************************************************************************************//**
				 * <summary> Returns a pointer to the effective load vector associated with the 
         *           numerical model.</summary>
				 *
				 * <returns> The relative pointer to the vector.</returns>
				 **************************************************************************************************/
				const axis::foundation::memory::RelativePointer GetReactionForcePointer(void) const;

				/**********************************************************************************************//**
				 * <summary> Returns a pointer to the effective load vector associated with the 
         *           numerical model.</summary>
				 *
				 * <returns> The relative pointer to the vector.</returns>
				 **************************************************************************************************/
				axis::foundation::memory::RelativePointer GetReactionForcePointer(void);

				/**********************************************************************************************//**
				 * <summary> Queries if the external load vector has been initialized.</summary>
				 *
				 * <returns> true if it was initialized, false otherwise.</returns>
				 **************************************************************************************************/
				bool IsExternalLoadAvailable(void) const;

				/**********************************************************************************************//**
				 * <summary> Queries if the internal force vector has been initialized.</summary>
				 *
				 * <returns> true if it was initialized, false otherwise.</returns>
				 **************************************************************************************************/
				bool IsInternalForceAvailable(void) const;

				/**********************************************************************************************//**
				 * <summary> Queries if the effective load vector has been initialized.</summary>
				 *
				 * <returns> true if it was initialized, false otherwise.</returns>
				 **************************************************************************************************/
				bool IsReactionForceAvailable(void) const;

        static axis::foundation::memory::RelativePointer Create(void);
      private:
        ModelDynamics(void);
        void *operator new(size_t bytes, void *ptr);
        void operator delete(void *, void *);
        axis::foundation::memory::RelativePointer _loads;
        axis::foundation::memory::RelativePointer _internalForces;
        axis::foundation::memory::RelativePointer _effectiveLoad;
			};
		}
	}
}


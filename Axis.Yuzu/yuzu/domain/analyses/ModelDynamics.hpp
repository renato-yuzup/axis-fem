#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace domain { namespace analyses {

/**********************************************************************************************//**
	* <summary> Contains dynamics information about the numerical model.</summary>
	**************************************************************************************************/
class ModelDynamics
{
public:
	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	GPU_ONLY ~ModelDynamics(void);

	/**********************************************************************************************//**
		* <summary> Resets this object.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetAll(void);

	/**********************************************************************************************//**
		* <summary> Resets the external loads of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetExternalLoad(void);

	/**********************************************************************************************//**
		* <summary> Resets the internal forces of the numerical model setting
		* 			 it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetInternalForce(void);

	/**********************************************************************************************//**
		* <summary> Resets the effective load of the numerical model setting
		* 			 it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetReactionForce(void);

	/**********************************************************************************************//**
		* <summary> Returns the external load vector associated with the numerical model.</summary>
		*
		* <returns> The load vector with as many positions as the number of dofs 
		* 			 in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& ExternalLoads(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the external load vector associated with the numerical model.</summary>
		*
		* <returns> The load vector with as many positions as the number of dofs 
		* 			 in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& ExternalLoads(void);

	/**********************************************************************************************//**
		* <summary> Returns the internal force vector associated with the numerical model.</summary>
		*
		* <returns> The internal force vector with as many positions as the number of dofs 
		* 			 in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& InternalForces(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the internal force vector associated with the numerical model.</summary>
		*
		* <returns> The internal force vector with as many positions as the number of dofs 
		* 			 in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& InternalForces(void);

	/**********************************************************************************************//**
		* <summary> Returns the effective load vector associated with the numerical model.</summary>
		*
		* <returns> The effective load vector with as many positions as the number of dofs 
		* 			 in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& ReactionForce(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the effective load vector associated with the numerical model.</summary>
		*
		* <returns> The effective load vector with as many positions as the number of dofs 
		* 			 in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& ReactionForce(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the the external load vector associated with the 
    *           numerical model.</summary>
		*
		* <returns> The relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetExternalLoadsPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the external load vector associated with the 
    *           numerical model.</summary>
		*
		* <returns> The relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetExternalLoadsPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the internal force vector associated with the 
    *           numerical model.</summary>
		*
		* <returns> The relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetInternalForcesPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the internal force vector associated with the 
    *           numerical model.</summary>
		*
		* <returns> The relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetInternalForcesPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the effective load vector associated with the 
    *           numerical model.</summary>
		*
		* <returns> The relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetReactionForcePointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the effective load vector associated with the 
    *           numerical model.</summary>
		*
		* <returns> The relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetReactionForcePointer(void);
private:
  GPU_ONLY ModelDynamics(void);

  axis::yuzu::foundation::memory::RelativePointer _loads;
  axis::yuzu::foundation::memory::RelativePointer _internalForces;
  axis::yuzu::foundation::memory::RelativePointer _effectiveLoad;
};

} } } } // namespace axis::yuzu::domain::analyses

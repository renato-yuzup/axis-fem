#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace domain { namespace analyses {

/**********************************************************************************************//**
	* <summary> Contains kinematic information about the numerical model.</summary>
	**************************************************************************************************/
class ModelKinematics
{
public:
	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	GPU_ONLY ~ModelKinematics(void);

	/**********************************************************************************************//**
		* <summary> Resets this object.</summary>
		*
		* <param name="numDofs"> Number of degrees of freedom.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetAll(void);

	/**********************************************************************************************//**
		* <summary> Resets the acceleration field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetAcceleration(void);

	/**********************************************************************************************//**
		* <summary> Resets the velocity field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetVelocity(void);

	/**********************************************************************************************//**
		* <summary> Resets the displacement field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetDisplacement(void);

	/**********************************************************************************************//**
		* <summary> Resets the displacement increment field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	GPU_ONLY void ResetDisplacementIncrement(void);

	/**********************************************************************************************//**
		* <summary> Returns the acceleration field of the numerical model.</summary>
		*
		* <returns> A vector representing the acceleration field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Acceleration(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the acceleration field of the numerical model.</summary>
		*
		* <returns> A vector representing the acceleration field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Acceleration(void);

	/**********************************************************************************************//**
		* <summary> Returns the velocity field of the numerical model.</summary>
		*
		* <returns> A vector representing the velocity field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Velocity(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the velocity field of the numerical model.</summary>
		*
		* <returns> A vector representing the velocity field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Velocity(void);

	/**********************************************************************************************//**
		* <summary> Returns the displacement field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Displacement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the displacement field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Displacement(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the acceleration field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetAccelerationPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the acceleration field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetAccelerationPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the velocity field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetVelocityPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the velocity field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetVelocityPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetDisplacementPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetDisplacementPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetDisplacementIncrementPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetDisplacementIncrementPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns the displacement increment field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& DisplacementIncrement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the displacement increment field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& DisplacementIncrement(void);
private:
	GPU_ONLY ModelKinematics(void);
  axis::yuzu::foundation::memory::RelativePointer displacement_;
  axis::yuzu::foundation::memory::RelativePointer displacementIncrement_;
  axis::yuzu::foundation::memory::RelativePointer velocity_;
  axis::yuzu::foundation::memory::RelativePointer acceleration_;
};			

} } } } // namespace axis::yuzu::domain::analyses

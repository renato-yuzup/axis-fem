#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace analyses {

/**********************************************************************************************//**
	* <summary> Contains kinematic information about the numerical model.</summary>
	**************************************************************************************************/
class AXISCOMMONLIBRARY_API ModelKinematics
{
public:

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	~ModelKinematics(void);

	/**********************************************************************************************//**
		* <summary> Resets this object.</summary>
		*
		* <param name="numDofs"> Number of degrees of freedom.</param>
		**************************************************************************************************/
	void ResetAll(size_type numDofs);

	/**********************************************************************************************//**
		* <summary> Resets the acceleration field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	void ResetAcceleration(size_type numDofs);

	/**********************************************************************************************//**
		* <summary> Resets the velocity field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	void ResetVelocity(size_type numDofs);

	/**********************************************************************************************//**
		* <summary> Resets the displacement field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	void ResetDisplacement(size_type numDofs);

	/**********************************************************************************************//**
		* <summary> Resets the displacement increment field of the numerical model
		* 			 setting it all to zero.</summary>
		*
		* <param name="numDofs"> Number of dofs in the model.</param>
		**************************************************************************************************/
	void ResetDisplacementIncrement(size_type numDofs);

	/**********************************************************************************************//**
		* <summary> Returns the acceleration field of the numerical model.</summary>
		*
		* <returns> A vector representing the acceleration field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& Acceleration(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the acceleration field of the numerical model.</summary>
		*
		* <returns> A vector representing the acceleration field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& Acceleration(void);

	/**********************************************************************************************//**
		* <summary> Returns the velocity field of the numerical model.</summary>
		*
		* <returns> A vector representing the velocity field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& Velocity(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the velocity field of the numerical model.</summary>
		*
		* <returns> A vector representing the velocity field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& Velocity(void);

	/**********************************************************************************************//**
		* <summary> Returns the displacement field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& Displacement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the displacement field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& Displacement(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the acceleration field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	const axis::foundation::memory::RelativePointer GetAccelerationPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the acceleration field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	axis::foundation::memory::RelativePointer GetAccelerationPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the velocity field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	const axis::foundation::memory::RelativePointer GetVelocityPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the velocity field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	axis::foundation::memory::RelativePointer GetVelocityPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	const axis::foundation::memory::RelativePointer GetDisplacementPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	axis::foundation::memory::RelativePointer GetDisplacementPointer(void);

	/**********************************************************************************************//**
		* <summary> Queries if the displacement field has been initialized.</summary>
		*
		* <returns> true if it was initialized, false otherwise.</returns>
		**************************************************************************************************/
	bool IsDisplacementFieldAvailable(void) const;
	/**********************************************************************************************//**
		* <summary> Queries if the displacement increment field has been initialized.</summary>
		*
		* <returns> true if it was initialized, false otherwise.</returns>
		**************************************************************************************************/
	bool IsDisplacementIncrementFieldAvailable(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	const axis::foundation::memory::RelativePointer GetDisplacementIncrementPointer(void) const;

	/**********************************************************************************************//**
		* <summary> Returns a pointer to the displacement field of the numerical model.</summary>
		*
		* <returns> A relative pointer to the vector.</returns>
		**************************************************************************************************/
	axis::foundation::memory::RelativePointer GetDisplacementIncrementPointer(void);

	/**********************************************************************************************//**
		* <summary> Returns the displacement increment field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& DisplacementIncrement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the displacement increment field of the numerical model.</summary>
		*
		* <returns> A vector representing the displacement field with as many positions
		* 			 as the number of dofs in the model.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& DisplacementIncrement(void);

	/**********************************************************************************************//**
		* <summary> Queries if the velocity field has been initialized.</summary>
		*
		* <returns> true if it was initialized, false otherwise.</returns>
		**************************************************************************************************/
	bool IsVelocityFieldAvailable(void) const;

	/**********************************************************************************************//**
		* <summary> Queries if the acceleration field has been initialized.</summary>
		*
		* <returns> true if it was initialized, false otherwise.</returns>
		**************************************************************************************************/
	bool IsAccelerationFieldAvailable(void) const;

  static axis::foundation::memory::RelativePointer Create(void);
private:
	ModelKinematics(void);
  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *, void *);
  axis::foundation::memory::RelativePointer displacement_;
  axis::foundation::memory::RelativePointer displacementIncrement_;
  axis::foundation::memory::RelativePointer velocity_;
  axis::foundation::memory::RelativePointer acceleration_;
};		

} } } // namespace axis::domain::analyses

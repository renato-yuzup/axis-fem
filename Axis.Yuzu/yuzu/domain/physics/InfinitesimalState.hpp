#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/blas/DenseMatrix.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace domain { namespace physics {

/**
 * Stores physical state of an infinitesimal point of a continuum body.
**/
class InfinitesimalState
{
public:
	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	GPU_ONLY ~InfinitesimalState(void);

	/**********************************************************************************************//**
		* <summary> Resets properties.</summary>
		**************************************************************************************************/
	GPU_ONLY void Reset(void);

	/**********************************************************************************************//**
		* <summary> Returns the current deformation gradient.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::DenseMatrix& DeformationGradient(void);

	/**********************************************************************************************//**
		* <summary> Returns the current deformation gradient.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::DenseMatrix& DeformationGradient(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the deformation gradient of the last computation step.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::DenseMatrix& LastDeformationGradient(void);

	/**********************************************************************************************//**
		* <summary> Returns the deformation gradient of the last computation step.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::DenseMatrix& LastDeformationGradient(void) const;


	/**********************************************************************************************//**
		* <summary> Returns the element strain tensor.</summary>
		*
		* <returns> A vector representing the strain tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Strain(void);

	/**********************************************************************************************//**
		* <summary> Returns the element strain tensor.</summary>
		*
		* <returns> A vector representing the strain tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Strain(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the last strain increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the strain increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& LastStrainIncrement(void);

	/**********************************************************************************************//**
		* <summary> Returns the last strain increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the strain increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& LastStrainIncrement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the element stress tensor.</summary>
		*
		* <returns> A vector representing the stress tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Stress(void);

	/**********************************************************************************************//**
		* <summary> Returns the element stress tensor.</summary>
		*
		* <returns> A vector representing the stress tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Stress(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the stress increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the stress increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& LastStressIncrement(void);

	/**********************************************************************************************//**
		* <summary> Returns the stress increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the stress increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& LastStressIncrement(void) const;

	/************************************************************************//**
		* <summary> Returns the last plastic strain increment to which this
    * material point was subjected.</summary>
		*
		* <returns> A vector representing the plastic strain increment tensor in 
    * Voigt notation.</returns>
		**************************************************************************/
  GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& PlasticStrain(void);

	/************************************************************************//**
		* <summary> Returns the last plastic strain increment to which this
    * material point was subjected.</summary>
		*
		* <returns> A vector representing the plastic strain increment tensor in 
    * Voigt notation.</returns>
		**************************************************************************/
  GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& PlasticStrain(void) const;

	/************************************************************************//**
		* <summary> Returns the effective plastic strain in this material point.
    * </summary>
		*
		* <returns> The effective plastic strain.</returns>
		**************************************************************************/
  GPU_ONLY real& EffectivePlasticStrain(void);

	/************************************************************************//**
		* <summary> Returns the effective plastic strain in this material point.
    * </summary>
		*
		* <returns> The effective plastic strain.</returns>
		**************************************************************************/
  GPU_ONLY real EffectivePlasticStrain(void) const;

  /**
   * Copies point state information from another instance.
   *
   * @param [in,out] source Source instance.
   */
  GPU_ONLY void CopyFrom(const InfinitesimalState& source);
private:
  axis::yuzu::foundation::memory::RelativePointer _strainIncrement;
  axis::yuzu::foundation::memory::RelativePointer _stressIncrement;
  axis::yuzu::foundation::memory::RelativePointer _strain;
  axis::yuzu::foundation::memory::RelativePointer _stress;
  axis::yuzu::foundation::memory::RelativePointer _curDeformationGradient;
  axis::yuzu::foundation::memory::RelativePointer _lastDeformationGradient;
  axis::yuzu::foundation::memory::RelativePointer plasticStrain_;
  real effectivePlasticStrain_;

  GPU_ONLY InfinitesimalState(void);
  InfinitesimalState(const InfinitesimalState&);
  InfinitesimalState& operator =(const InfinitesimalState&);
};

} } } } // namespace axis::yuzu::domain::physics


#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/fwd/finite_element_fwd.hpp"
#include "foundation/blas/blas.hpp"

namespace axis { namespace domain { namespace physics {

/**
 * Stores physical state of an infinitesimal point of a continuum body.
**/
class AXISCOMMONLIBRARY_API InfinitesimalState
{
public:
	/**********************************************************************************************//**
		* <summary> Constructor.</summary>
		**************************************************************************************************/
	InfinitesimalState(void);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	~InfinitesimalState(void);

	/**********************************************************************************************//**
		* <summary> Destroys this object.</summary>
		**************************************************************************************************/
	void Destroy(void) const;

	/**********************************************************************************************//**
		* <summary> Resets properties.</summary>
		**************************************************************************************************/
	void Reset(void);

	/**********************************************************************************************//**
		* <summary> Returns the current deformation gradient.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	axis::foundation::blas::DenseMatrix& DeformationGradient(void);

	/**********************************************************************************************//**
		* <summary> Returns the current deformation gradient.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::DenseMatrix& DeformationGradient(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the deformation gradient of the last computation step.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	axis::foundation::blas::DenseMatrix& LastDeformationGradient(void);

	/**********************************************************************************************//**
		* <summary> Returns the deformation gradient of the last computation step.</summary>
		*
		* <returns> A matrix representing the deformation gradient tensor.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::DenseMatrix& LastDeformationGradient(void) const;


	/**********************************************************************************************//**
		* <summary> Returns the element strain tensor.</summary>
		*
		* <returns> A vector representing the strain tensor in Voigt notation.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& Strain(void);

	/**********************************************************************************************//**
		* <summary> Returns the element strain tensor.</summary>
		*
		* <returns> A vector representing the strain tensor in Voigt notation.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& Strain(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the last strain increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the strain increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& LastStrainIncrement(void);

	/**********************************************************************************************//**
		* <summary> Returns the last strain increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the strain increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& LastStrainIncrement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the element stress tensor.</summary>
		*
		* <returns> A vector representing the stress tensor in Voigt notation.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& Stress(void);

	/**********************************************************************************************//**
		* <summary> Returns the element stress tensor.</summary>
		*
		* <returns> A vector representing the stress tensor in Voigt notation.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& Stress(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the stress increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the stress increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	axis::foundation::blas::ColumnVector& LastStressIncrement(void);

	/**********************************************************************************************//**
		* <summary> Returns the stress increment to which the element is being subjected.</summary>
		*
		* <returns> A vector representing the stress increment tensor in Voigt notation.</returns>
		**************************************************************************************************/
	const axis::foundation::blas::ColumnVector& LastStressIncrement(void) const;

	/************************************************************************//**
		* <summary> Returns the last plastic strain increment to which this
    * material point was subjected.</summary>
		*
		* <returns> A vector representing the plastic strain increment tensor in 
    * Voigt notation.</returns>
		**************************************************************************/
  axis::foundation::blas::ColumnVector& PlasticStrain(void);

	/************************************************************************//**
		* <summary> Returns the last plastic strain increment to which this
    * material point was subjected.</summary>
		*
		* <returns> A vector representing the plastic strain increment tensor in 
    * Voigt notation.</returns>
		**************************************************************************/
  const axis::foundation::blas::ColumnVector& PlasticStrain(void) const;

	/************************************************************************//**
		* <summary> Returns the effective plastic strain in this material point.
    * </summary>
		*
		* <returns> The effective plastic strain.</returns>
		**************************************************************************/
  real& EffectivePlasticStrain(void);

	/************************************************************************//**
		* <summary> Returns the effective plastic strain in this material point.
    * </summary>
		*
		* <returns> The effective plastic strain.</returns>
		**************************************************************************/
  real EffectivePlasticStrain(void) const;

  /**
   * Copies point state information from another instance.
   *
   * @param [in,out] source Source instance.
   */
  void CopyFrom(const InfinitesimalState& source);

  static axis::foundation::memory::RelativePointer Create(void);
  void *operator new(size_t bytes);
  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *);
  void operator delete(void *, void *);
private:
  axis::foundation::memory::RelativePointer _strainIncrement;
  axis::foundation::memory::RelativePointer _stressIncrement;
  axis::foundation::memory::RelativePointer _strain;
  axis::foundation::memory::RelativePointer _stress;
  axis::foundation::memory::RelativePointer _curDeformationGradient;
  axis::foundation::memory::RelativePointer _lastDeformationGradient;
  axis::foundation::memory::RelativePointer plasticStrain_;
  real effectivePlasticStrain_;
};

} } }


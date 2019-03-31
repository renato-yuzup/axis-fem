/// <summary>
/// Contains definition for the abstract class 
/// axis::domain::materials::Base::MaterialModel.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/fwd/finite_element_fwd.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "foundation/blas/blas.hpp"
#include "MaterialStrategy.hpp"

namespace axis { namespace domain { namespace materials {

/**
 * Represents the constitutive model associated to the material 
 * of a point in the continuum.
**/
class AXISCOMMONLIBRARY_API MaterialModel
{
public:

  /**
   * Constructor.
   *
   * @param density   The linear density.
   * @param numPoints Number of material points this material model instance will
   *                  work on.
   */
	MaterialModel(real density, int numPoints);

  /**
   * Destructor.
  **/
	virtual ~MaterialModel(void);

  /**
   * Destroys this object.
  **/
	virtual void Destroy(void) const = 0;

  /**
   * Returns the material tensor.
   *
   * @return A matrix representing the material stiffness..
  **/
	virtual const axis::foundation::blas::DenseMatrix& 
    GetMaterialTensor(void) const = 0;

  /**
   * Makes a deep copy of this object.
   *
   * @return A copy of this object.
  **/

  /**
   * Makes a deep copy of this instance.
   *
   * @param numPoints Number of material points the material model instance
   *                  will work on.
   *
   * @return A copy of this instance.
   */
	virtual MaterialModel& Clone(int numPoints) const = 0;

  /**
    * 
    *
  **/

  /**
   * Calculates and updates physical state of an infinitesimal point.
   *
   * @param [in,out] updatedState Updated state of the infinitesimal point.
   * @param currentState          The current state of the infinitesimal point.
   * @param timeInfo              Information describing the time state of the 
   *                              analysis.
   * @param materialPointIndex    Zero-based index of the material point to 
   *                              which stress state is being updated.
   */
	virtual void UpdateStresses(
    axis::domain::physics::UpdatedPhysicalState& updatedState,
    const axis::domain::physics::InfinitesimalState& currentState,
		const axis::domain::analyses::AnalysisTimeline& timeInfo,
    int materialPointIndex) = 0;

  /**
   * Returns the material density.
   *
   * @return The density.
  **/
	real Density(void) const;

  /**
   * Returns the material bulk modulus, the substance 
   * resistance to uniform (hydrostatic) stress.
   *
   * @return The bulk modulus.
  **/
  virtual real GetBulkModulus(void) const = 0;

  /**
   * Returns the material shear modulus, the ratio of shear 
   * strain to shear stress.
   *
   * @return The shear modulus.
  **/
  virtual real GetShearModulus(void) const = 0;

  /**
   * Returns the velocity of propagation of mechanical waves in this material.
   *
   * @return The wave propagation speed.
   */
	virtual real GetWavePropagationSpeed(void) const = 0;

  /**
   * Returns if this material model can be processed in a GPU environment.
   *
   * @return true if GPU capable, false otherwise.
  **/
  virtual bool IsGPUCapable(void) const;

  /**
   * Returns if this material model can be processed in a CPU environment.
   *
   * @return true if CPU capable, false otherwise.
  **/
  virtual bool IsCPUCapable(void) const;

  /**
   * Returns the unique identifier for this material model type.
   *
   * @return An universal unique identifier (UUID) which is exclusive among
   *         all material model types.
  **/
  virtual axis::foundation::uuids::Uuid GetTypeId(void) const = 0;

  virtual size_type GetDataBlockSize(void) const;

  /**
   * Initializes GPU material data.
   *
   * @param [in,out] baseDataAddress Base memory address where formulation data 
   *                                 should be written.
  **/
  virtual void InitializeGPUData(void *baseDataAddress, real *density, 
    real *waveSpeed, real *bulkModulus, real *shearModulus, 
    real *materialTensor);

  virtual MaterialStrategy& GetGPUStrategy(void);
  void *operator new(size_t bytes);
  void operator delete(void *ptr);
protected:
  int GetMaterialPointCount(void) const;
private:
  real density_;
  int pointCount_;
};			

} } } // namespace axis::domain::materials

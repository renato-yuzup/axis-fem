/// <summary>
/// Contains definition for the class axis::domain::materials::LinearElasticIsotropicModel.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "domain/materials/MaterialModel.hpp"

namespace axis { namespace domain { namespace materials {

/// <summary>
/// Implements the behavior of a linear elastic and isotropic material.
/// </summary>
class LinearElasticIsotropicModel : public axis::domain::materials::MaterialModel
{
public:
  /**
   * Constructor.
   *
   * @param youngModulus       Young's Modulus (elastic modulus).
   * @param poissonCoefficient Poisson coefficient.
   * @param density            The linear density.
   */
	LinearElasticIsotropicModel(real youngModulus, real poissonCoefficient, 
    real density);

  /**
   * Destroys this object.
   */
	virtual ~LinearElasticIsotropicModel(void);
	virtual void Destroy( void ) const;
	virtual axis::domain::materials::MaterialModel& Clone( int numPoints ) const;
	virtual const axis::foundation::blas::DenseMatrix& 
    GetMaterialTensor( void ) const;
	virtual void UpdateStresses( 
    axis::domain::physics::UpdatedPhysicalState& updatedState,
    const axis::domain::physics::InfinitesimalState& currentState, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo,
    int materialPointIndex);
  virtual real GetBulkModulus(void) const;
  virtual real GetShearModulus(void) const;
	virtual real GetWavePropagationSpeed( void ) const;
  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;
  virtual bool IsGPUCapable( void ) const;
  virtual size_type GetDataBlockSize( void ) const;
  virtual void InitializeGPUData( void *baseDataAddress, real *density, 
    real *waveSpeed, real *bulkModulus, real *shearModulus, 
    real *materialTensor );
  virtual MaterialStrategy& GetGPUStrategy( void );
private:
  real youngModulus_;
  real poisson_;
  axis::foundation::memory::RelativePointer materialMatrix_;
  real waveVelocity_;
};			

} } } // namespace axis::domain::materials

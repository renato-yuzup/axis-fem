/// <summary>
/// Contains definition for the class axis::domain::materials::NeoHookeanModel.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "domain/materials/MaterialModel.hpp"

namespace axis { namespace domain { namespace materials {

/// <summary>
/// Describes the behavior of rubber-like materials by using Neohookean 
/// constitutive relations.
/// </summary>
class NeoHookeanModel : public axis::domain::materials::MaterialModel
{
public:

	/**************************************************************************************************
		* <summary>	Creates a new instance of this class. </summary>
		*
		* <param name="youngModulus">			 of the material. </param>
		* <param name="poissonCoefficient">	 of the material. </param>
		* <param name="density">				The material density. </param>
		**************************************************************************************************/

  /**
   * Constructor.
   *
   * @param youngModulus       Young's Modulus (elastic modulus).
   * @param poissonCoefficient Poisson coefficient.
   * @param density            The linear material density.
   * @param numPoints          Number of points this instance will work on.
   */
	NeoHookeanModel(real youngModulus, real poissonCoefficient, real density);

	virtual ~NeoHookeanModel(void);

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
  real lambdaLameCoeff_;
  real muLameCoeff_;
  real elasticModulus_;
  real poisson_;
  axis::foundation::memory::RelativePointer materialMatrix_;
  real waveVelocity_;
};			

} } } // namespace axis::domain::materials

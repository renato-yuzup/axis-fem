#pragma once
#include "domain/materials/MaterialModel.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace materials {

/**
 * Defines a simple isotropic, bi-linear elasto-plastic constitutive model,
 * with isotropic hardening.
 */
class BiLinearPlasticityModel : public MaterialModel
{
public:
  BiLinearPlasticityModel(real elasticModulus, real poissonCoefficient,
    real yieldStress, real hardeningCoefficient, real density, int numPoints);
  virtual ~BiLinearPlasticityModel(void);
  virtual void Destroy( void ) const;
  virtual const axis::foundation::blas::DenseMatrix& 
    GetMaterialTensor( void ) const;
  virtual MaterialModel& Clone( int numPoints ) const;
  virtual void UpdateStresses( 
    axis::domain::physics::UpdatedPhysicalState& updatedState, 
    const axis::domain::physics::InfinitesimalState& currentState, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo, 
    int materialPointIndex );
  virtual real GetBulkModulus( void ) const;
  virtual real GetShearModulus( void ) const;
  virtual real GetWavePropagationSpeed( void ) const;
  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;
  virtual bool IsGPUCapable( void ) const;
  virtual size_type GetDataBlockSize( void ) const;
  virtual void InitializeGPUData( void *baseDataAddress, real *density, 
    real *waveSpeed, real *bulkModulus, real *shearModulus, real *materialTensor);
  virtual MaterialStrategy& GetGPUStrategy( void );
private:
  void UpdateElasticStress( 
    axis::domain::physics::UpdatedPhysicalState& updatedState, 
    const axis::domain::physics::InfinitesimalState& currentState, 
    const axis::foundation::blas::SymmetricMatrix& rateOfDeformationTensor,
    const axis::foundation::blas::DenseMatrix& spinTensor,
    real timeIncrement);
  void ExecutePlasticCorrection( 
    axis::domain::physics::UpdatedPhysicalState& updatedState, 
    const axis::domain::physics::InfinitesimalState& currentState,
    axis::foundation::blas::SymmetricMatrix& rateOfElasticDeformationTensor,
    const axis::foundation::blas::DenseMatrix& spinTensor,
    real timeIncrement, int materialPointIndex);
  void CalculateStressRate(
    axis::foundation::blas::SymmetricMatrix& stressRate,
    const axis::foundation::blas::DenseMatrix& elasticityMatrix,
    const axis::foundation::blas::SymmetricMatrix& stressTensor,
    const axis::foundation::blas::SymmetricMatrix& rateOfDeformationTensor,
    const axis::foundation::blas::DenseMatrix& spinTensor);

  real elasticModulus_, poisson_;
  real yieldStress_;
  real hardeningCoeff_;
  real waveVelocity_;
  axis::foundation::memory::RelativePointer materialTensor_;
  axis::foundation::memory::RelativePointer flowStressPtr_;
  real *flowStress_;
};

} } } // namespace axis::domain::materials

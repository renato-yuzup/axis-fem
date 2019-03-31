#if defined _DEBUG || defined DEBUG
#pragma once
#include "domain/materials/MaterialModel.hpp"
#include "foundation/blas/DenseMatrix.hpp"

namespace axis { namespace unit_tests { namespace standard_elements {

class MockLinearMaterial : public axis::domain::materials::MaterialModel
{
public:
  MockLinearMaterial(real poisson);
  ~MockLinearMaterial(void);
  virtual void Destroy( void ) const;
  virtual const axis::foundation::blas::DenseMatrix& 
    GetMaterialTensor(void) const;
  virtual MaterialModel& Clone( int numPoints ) const;
  virtual void UpdateStresses( 
    axis::domain::physics::UpdatedPhysicalState& updatedState, 
    const axis::domain::physics::InfinitesimalState& currentState, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo,
    int materialPointIndex);
  virtual real GetBulkModulus( void ) const;
  virtual real GetShearModulus( void ) const;
  virtual real GetWavePropagationSpeed( void ) const;
  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;
private:
  axis::foundation::blas::DenseMatrix tensor_;
  real waveVelocity_;
  real shearModulus_, bulkModulus_;
  real poisson_;
};

} } } // namespace axis::unit_tests::standard_elements

#endif

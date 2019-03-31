#if defined _DEBUG || defined DEBUG

#include "stdafx.h"
#include "MockLinearMaterial.hpp"
#include <exception>
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "domain/physics/InfinitesimalState.hpp"

namespace aus = axis::unit_tests::standard_elements;
namespace afb = axis::foundation::blas;
namespace ada = axis::domain::analyses;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;

#define E     200e9
#define RHO   7800

aus::MockLinearMaterial::MockLinearMaterial( real poisson ) :
MaterialModel(RHO, 1), tensor_(6,6)
{
  // create material matrix
  axis::foundation::blas::DenseMatrix& D = tensor_;
  D.ClearAll();

  // matrix terms
  real c11 = E*(1-poisson) / ((1-2*poisson)*(1+poisson));
  real c12 = E*poisson / ((1-2*poisson)*(1+poisson));
  real G  = E / (2*(1+poisson));

  D(0,0) = c11;		D(0,1) = c12;		D(0,2) = c12                                         ;
  D(1,0) = c12;		D(1,1) = c11;		D(1,2) = c12                                         ;
  D(2,0) = c12;		D(2,1) = c12;		D(2,2) = c11                                         ;
                                                   D(3,3) = G                          ;      
                                                               D(4,4) = G              ;      
                                                                           D(5,5) = G  ;      
  waveVelocity_ = sqrt(E * (1-poisson) / (RHO * (1+poisson)*(1-2*poisson)));
  shearModulus_ = G;
  bulkModulus_ = E / (3 * (1-2*poisson));
  poisson_ = poisson;
}

aus::MockLinearMaterial::~MockLinearMaterial( void )
{
  // nothing to do here
}

void aus::MockLinearMaterial::Destroy( void ) const
{
  delete this;
}

const afb::DenseMatrix& aus::MockLinearMaterial::GetMaterialTensor( void ) const
{
  return tensor_;
}

adm::MaterialModel& aus::MockLinearMaterial::Clone( int ) const
{
  return *new MockLinearMaterial(poisson_);
}

void aus::MockLinearMaterial::UpdateStresses( 
  adp::UpdatedPhysicalState& updatedState, 
  const adp::InfinitesimalState& currentState, const ada::AnalysisTimeline&, int)
{
  afb::ColumnVector& s = updatedState.Stress();
  afb::ColumnVector& ds = updatedState.LastStressIncrement();
  const afb::ColumnVector& de = currentState.LastStrainIncrement();
  const afb::DenseMatrix& D = tensor_;
  afb::VectorProduct(ds, 1.0, D, de);
  afb::VectorSum(s, 1.0, s, 1.0, ds);
}

real aus::MockLinearMaterial::GetBulkModulus( void ) const
{
  return bulkModulus_;
}

real aus::MockLinearMaterial::GetShearModulus( void ) const
{
  return shearModulus_;
}

real aus::MockLinearMaterial::GetWavePropagationSpeed( void ) const
{
  return waveVelocity_;
}

axis::foundation::uuids::Uuid axis::unit_tests::standard_elements::MockLinearMaterial::GetTypeId( void ) const
{
  throw std::exception("The method or operation is not implemented.");
}

#endif

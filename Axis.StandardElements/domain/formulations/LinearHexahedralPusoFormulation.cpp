#define NOMINMAX
#include "LinearHexahedralPusoFormulation.hpp"
#include <assert.h>
#include <boost/detail/limits.hpp>
#include "LinearHexahedralPusoFormulationHelper.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/elements/Node.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "foundation/NotImplementedException.hpp"
#include "foundation/NotSupportedException.hpp"
#include "Foundation/BLAS/ColumnVector.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"
#include "foundation/memory/pointer.hpp"

namespace adf = axis::domain::formulations;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace af = axis::foundation;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

namespace {
  typedef adf::LinearHexahedralPusoFormulation hexa;
}

hexa::LinearHexahedralPusoFormulation(void)
{
  lumpedMass_ = NULLPTR;
  Bmatrix_ = NULLPTR;
  nodeCoordinates_ = NULLPTR;
  jacobian_ = NULLPTR;
  for (int i = 0; i < 4; i++)
  {
    hourglassForces_[i] = NULLPTR;
  }  
  volume_ = 0;
  hourglassEnergy_ = 0;
}

hexa::~LinearHexahedralPusoFormulation(void)
{
  ClearMemory();
  if (nodeCoordinates_ != NULLPTR)
  {
    absref<afb::DenseMatrix>(nodeCoordinates_).Destroy();
    System::ModelMemory().Deallocate(nodeCoordinates_);
  }
  if (Bmatrix_ != NULLPTR)
  {
    absref<afb::DenseMatrix>(Bmatrix_).Destroy();
    System::ModelMemory().Deallocate(Bmatrix_);
  }
  if (jacobian_ != NULLPTR)
  {
    absref<afb::DenseMatrix>(jacobian_).Destroy();
    System::ModelMemory().Deallocate(jacobian_);
  }
  for (int i = 0; i < 4; i++)
  {
    if (hourglassForces_[i] != NULLPTR)
    {
      absref<afb::ColumnVector>(hourglassForces_[i]).Destroy();
      System::ModelMemory().Deallocate(hourglassForces_[i]);
      hourglassForces_[i] = NULLPTR;
    }
  }  
  jacobian_ = NULLPTR;
  Bmatrix_ = NULLPTR;
  nodeCoordinates_ = NULLPTR;
}

void hexa::Destroy( void ) const
{
  delete this;
}

const afb::SymmetricMatrix& hexa::GetStiffness( void ) const
{
  // TODO : implement stiffness matrix according to Puso (2000).
  throw af::NotImplementedException(
    _T("This element formulation does not implement stiffness matrix."));
}

const afb::SymmetricMatrix& hexa::GetConsistentMass( void ) const
{
  throw af::NotImplementedException(
    _T("This element formulation does not implement consistent mass."));
}

const afb::ColumnVector& hexa::GetLumpedMass( void ) const
{
  return absref<afb::ColumnVector>(lumpedMass_);
}

void hexa::AllocateMemory( void )
{
  // initialize matrices
  nodeCoordinates_ = afb::DenseMatrix::Create(8, 3);
  Bmatrix_ = afb::DenseMatrix::Create(3, 8);
  jacobian_ = afb::DenseMatrix::Create(3, 3);
  for (int i = 0; i < 4; i++)
  {
    hourglassForces_[i] = afb::ColumnVector::Create(3);
  }  
}

void hexa::CalculateInitialState( void )
{
  // get initial nodal coordinates
  ade::ElementGeometry& geometry = Element().Geometry();
  auto& Bmatrix = absref<afb::DenseMatrix>(Bmatrix_);
  auto& nodeCoordinates = absref<afb::DenseMatrix>(nodeCoordinates_);
  auto& jacobian = absref<afb::DenseMatrix>(jacobian_);
  CalculateNodeCoordinate(nodeCoordinates, geometry);
  CalculateBMatrix(Bmatrix, volume_, nodeCoordinates);
  CalculateJacobian(jacobian, nodeCoordinates);

  // clear hourglass force vectors
  for (int i = 0; i < 4; i++)
  {
    absref<afb::ColumnVector>(hourglassForces_[i]).ClearAll();
  }
}

void hexa::UpdateStrain( const afb::ColumnVector& elementDisplacementIncrement )
{
  const afb::ColumnVector& du = elementDisplacementIncrement; // shorthand
  const afb::DenseMatrix& B = absref<afb::DenseMatrix>(Bmatrix_);
  afb::ColumnVector& strain = Element().PhysicalState().Strain();
  afb::ColumnVector& strainIncrement = Element().PhysicalState().LastStrainIncrement();

  // our B-matrix is somewhat "different", so we rearrange the matrix-vector 
  // computation normally done
  strainIncrement.ClearAll();
  for (int i = 0; i < 8; i++)
  {
    real e1 = B(0,i)*du(3*i);                                 // du1/dx
    real e2 = B(1,i)*du(3*i + 1);                             // du2/dy
    real e3 = B(2,i)*du(3*i + 2);                             // du3/dz
    real e4 = B(2,i)*du(3*i + 1) + B(1,i)*du(3*i + 2);        // du2/dz + du3/dy
    real e5 = B(2,i)*du(3*i) + B(0,i)*du(3*i + 2);            // du1/dz + du3/dx
    real e6 = B(1,i)*du(3*i) + B(0,i)*du(3*i + 1);            // du1/dy + du2/dx
    strainIncrement(0) += e1;
    strainIncrement(1) += e2;
    strainIncrement(2) += e3;
    strainIncrement(3) += e4;
    strainIncrement(4) += e5;
    strainIncrement(5) += e6;
  }
  afb::VectorSum(strain, 1.0, strain, 1.0, strainIncrement);
}

void hexa::UpdateInternalForce( afb::ColumnVector& internalForce, 
  const afb::ColumnVector& elementDisplacementIncrement, 
  const afb::ColumnVector& elementVelocity, 
  const ada::AnalysisTimeline& timeInfo )
{
  internalForce.ClearAll();
  CalculateCentroidalInternalForce(internalForce);
  StabilizeInternalForce(internalForce, elementDisplacementIncrement);
  internalForce.Scale(-1.0);
}

void hexa::UpdateMatrices( const ade::MatrixOption& whichMatrices, 
  const afb::ColumnVector& elementDisplacement, 
  const afb::ColumnVector& elementVelocity )
{
  if (whichMatrices.DoesRequestConsistentMassMatrix())
  {
    throw af::NotSupportedException(
      _T("This element formulation does not support consistent mass."));
  }
  else if (whichMatrices.DoesRequestLumpedMassMatrix())
  {
    UpdateLumpedMassMatrix();
  }
  else if (whichMatrices.DoesRequestStiffnessMatrix())
  {
    throw af::NotSupportedException(
      _T("This element formulation does not support stiffness matrix."));
  }
  else
  {
    assert(!_T("An unpredictable matrix option was requested!"));
  }
}

void hexa::ClearMemory( void )
{
  if (lumpedMass_ != NULLPTR)
  {
    absref<afb::ColumnVector>(lumpedMass_).Destroy();
    System::ModelMemory().Deallocate(lumpedMass_);
  }
  lumpedMass_ = NULLPTR;
}

real hexa::GetCriticalTimestep(const afb::ColumnVector& modelDisplacement) const
{
  /* According to Puso (2000), the calculation below is empirically faster and
     works for most of the problems. Also, other more conservative equations, 
     is generally coupled to material model and does not account hourglass
     stiffness, which generally presents two significant drawbacks.
  **/
  const ade::ElementGeometry& geometry = Element().Geometry();
  const adm::MaterialModel& material = Element().Material();
  const real speed = material.GetWavePropagationSpeed();

  // get the shortest distance between nodes
  real length = std::numeric_limits<real>::max();
  for (int i = 0; i < 7; i++)
  {
    const ade::Node& nodeX = geometry[i];
    for (int j = i + 1; j < 8; j++)
    {
      const ade::Node& nodeY = geometry[j];
      real dx = nodeX.X() - nodeY.X();
      real dy = nodeX.Y() - nodeY.Y();
      real dz = nodeX.Z() - nodeY.Z();
      real dist = sqrt(dx*dx + dy*dy + dz*dz);
      if (dist < length)
      {
        length = dist;
      }
    }
  }
  
  // this is a pessimistic approach: the fastest speed and the shortest distance
  real solidLongitudinalWaveSpeed = sqrt(speed / length);
  return solidLongitudinalWaveSpeed;
}

void hexa::UpdateLumpedMassMatrix( void )
{
  if (lumpedMass_ == NULLPTR)
  {
    lumpedMass_ = axis::foundation::blas::ColumnVector::Create(24);
  }
  adm::MaterialModel& material = Element().Material();
  real massPerNode = volume_ * material.Density() / 8;
  absref<afb::ColumnVector>(lumpedMass_).SetAll(massPerNode);
}

void hexa::CalculateCentroidalInternalForce( afb::ColumnVector& internalForce )
{
  const afb::DenseMatrix& B = absref<afb::DenseMatrix>(Bmatrix_);
  const afb::ColumnVector& stress = Element().PhysicalState().Stress();
  afb::ColumnVector& fint = internalForce;
  for (int i = 0; i < 8; i++)
  {
    real f1 = B(0, i)*stress(0) + B(2, i)*stress(4) + B(1, i)*stress(5);
    real f2 = B(1, i)*stress(1) + B(2, i)*stress(3) + B(0, i)*stress(5);
    real f3 = B(2, i)*stress(2) + B(1, i)*stress(3) + B(0, i)*stress(4);

    fint(3*i  ) = f1;
    fint(3*i+1) = f2;
    fint(3*i+2) = f3;
  }
  fint.Scale(volume_);
}

real adf::LinearHexahedralPusoFormulation::GetTotalArtificialEnergy( void ) const
{
  return hourglassEnergy_;
}

void hexa::StabilizeInternalForce( axis::foundation::blas::ColumnVector& internalForce, 
                                   const axis::foundation::blas::ColumnVector& displacementIncrement )
{
  id_type id = Element().GetInternalId();

  // These are some shorthands that we are going to use
  const afb::DenseMatrix& J = absref<afb::DenseMatrix>(jacobian_);
  const afb::DenseMatrix& B = absref<afb::DenseMatrix>(Bmatrix_);
  const afb::ColumnVector& du = displacementIncrement;
  afb::DenseMatrix& X = absref<afb::DenseMatrix>(nodeCoordinates_);
  const afb::DenseMatrix& C = Element().Material().GetMaterialTensor();

  afb::ColumnVector sv1(8), sv2(8), sv3(8), sv4(8);
  afb::ColumnVector *stabilizationVector[4] = {&sv1, &sv2, &sv3, &sv4};
  for (int i = 0; i < 4; i++) // iterate through vectors
  {
    // gamma_i = h_i
    for (int idx = 0; idx < 8; idx++)
    {
      (*stabilizationVector[i])(idx) = hourglassVector_[i][idx];
    }    

    // sum [j = 1 : 3]
    for (int j = 0; j < 3; j++)
    {
      // dot product h_i * x_j
      real dotProduct = 0;
      for (int m = 0; m < 8; m++)
      {
        dotProduct += hourglassVector_[i][m] * X(m, j);
      }

      for (int k = 0; k < 8; k++) // iterate through vector indices
      {
        (*stabilizationVector[i])(k) -= dotProduct * B(j, k);
      }
    }

    stabilizationVector[i]->Scale(0.125);
  }

  // Step 2: Calculate hourglass displacements (already in isoparametric 
  // domain, Eq. 8)
  afb::ColumnVector v1(8), v2(8), v3(8), v4(8);
  afb::ColumnVector *hourglassDisplacement[4] = {&v1, &v2, &v3, &v4};
  for (int i = 0; i < 4; i++)
  {
    hourglassDisplacement[i]->ClearAll();
    real v_1 = 0, v_2 = 0, v_3 = 0;
    for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
    {
      // in generalized coordinates...
      v_1 += (*stabilizationVector[i])(nodeIdx) * du(3*nodeIdx    );
      v_2 += (*stabilizationVector[i])(nodeIdx) * du(3*nodeIdx + 1);
      v_3 += (*stabilizationVector[i])(nodeIdx) * du(3*nodeIdx + 2);
    }

    // ...now in isoparametric coordinates
    for (int k = 0; k < 3; k++)
    {
      (*hourglassDisplacement[i])(k) = J(0, k)*v_1 + J(1, k)*v_2 + J(2, k)*v_3;
    }
  }

  // Step 3: Calculate hat(J)_0^(-1), which maps strains to isoparametric 
  // coordinates (Eq. 3.61 pg. 3.14, LS-DYNA Theory Manual)
  afb::ColumnVector strainJacobian(6);  
  real jSquaredNorm[3];
  for (int i = 0; i < 3; i++)
  {
    real squaredNorm = J(0,i)*J(0,i) + J(1,i)*J(1,i) + J(2,i)*J(2,i);
    jSquaredNorm[i] = squaredNorm;
  }
  strainJacobian(0) = 1.0 / jSquaredNorm[0];
  strainJacobian(1) = 1.0 / jSquaredNorm[1];
  strainJacobian(2) = 1.0 / jSquaredNorm[2];
  strainJacobian(3) = 1.0 / sqrt(jSquaredNorm[0] * jSquaredNorm[1]);
  strainJacobian(4) = 1.0 / sqrt(jSquaredNorm[2] * jSquaredNorm[1]);
  strainJacobian(5) = 1.0 / sqrt(jSquaredNorm[0] * jSquaredNorm[2]);

  // Step 4: Map material stiffness matrix to isoparametric space.
  afb::DenseMatrix Ciso(6, 6);
  for (int i = 0; i < 6; i++)
  {
    for (int j = 0; j < 6; j++)
    {
      real c_ij = strainJacobian(i) * C(i, j) * strainJacobian(j);
      Ciso(i, j) = c_ij;
    }
  }

  // Step 5: Calculate internal degrees of freedom for enhanced strain.
  real H = Ciso(0,0) + Ciso(1,1) + Ciso(2,2) + 2.0*(Ciso(0,1) + 
    Ciso(1,2) + Ciso(0,2));
  real da1 = 1.0/Ciso(0,0) * (Ciso(0,1)*v1(1) + Ciso(0,4)*v2(1) + 
    Ciso(0,2)*v2(2) + Ciso(0,4)*v1(2));
  real da2 = 1.0/Ciso(1,1) * (Ciso(1,0)*v1(0) + Ciso(1,5)*v3(0) + 
    Ciso(1,2)*v3(2));
  real da3 = 1.0/Ciso(2,2) * (Ciso(2,0)*v2(0) + Ciso(2,3)*v3(0) + 
    Ciso(2,1)*v3(1) + Ciso(2,3)*v2(1));
  real da4 = v4(2)/H * (Ciso(0,2) + Ciso(1,2) + Ciso(2,2));
  real da5 = v4(0)/H * (Ciso(0,0) + Ciso(1,0) + Ciso(2,0));
  real da6 = v4(1)/H * (Ciso(0,1) + Ciso(1,1) + Ciso(2,1));

  // Step 6: Calculate hourglass force increment
  afb::ColumnVector& df1 = absref<afb::ColumnVector>(hourglassForces_[0]);
  afb::ColumnVector& df2 = absref<afb::ColumnVector>(hourglassForces_[1]);
  afb::ColumnVector& df3 = absref<afb::ColumnVector>(hourglassForces_[2]);
  afb::ColumnVector& df4 = absref<afb::ColumnVector>(hourglassForces_[3]);
  real beta1 = 8.0/3.0 * volume_;
  real beta2 = 8.0/9.0 * volume_;
  real df11 = beta1 * (Ciso(0,0)*v1(0) + Ciso(0,2)*v3(2) + Ciso(0,5)*(v3(0) + 
    v1(2)) - Ciso(1,0)*da2);
  real df12 = beta1 * (Ciso(1,1)*v1(1) + Ciso(1,2)*v2(2) + Ciso(1,4)*(v2(1) + 
    v1(2)) - Ciso(0,1)*da1);
  real df13 = beta1 * (Ciso(4,4)*(v2(1) + v1(2)) + Ciso(5,5)*(v3(0) + v1(2)) + 
    Ciso(5,0)*v1(0) + Ciso(4,1)*v1(1)
    + Ciso(4,2)*v2(2) + Ciso(5,2)*v3(2) - Ciso(0,4)*da1);
  real df21 = beta1 * (Ciso(0,0)*v2(0) + Ciso(0,1)*v3(1) + Ciso(0,3)*(v3(0) + 
    v2(1)) - Ciso(2,0)*da3);
  real df22 = beta1 * (Ciso(3,3)*(v3(0) + v2(1)) + Ciso(4,4)*(v2(1) + v1(2)) + 
    Ciso(3,0)*v2(0) + Ciso(3,1)*v3(1) 
    + Ciso(4,1)*v1(1) + Ciso(4,2)*v2(2) - Ciso(0,4)*da1 - Ciso(2,3)*da3);
  real df23 = beta1 * (Ciso(2,1)*v1(1) + Ciso(2,2)*v2(2) + Ciso(2,4)*(v2(1) + 
    v1(2)) - Ciso(0,2)*da1);
  real df31 = beta1 * (Ciso(3,3)*(v3(0) + v2(1)) + Ciso(5,5)*(v3(0) + v1(2)) + 
    Ciso(3,0)*v2(0) + Ciso(3,1)*v3(1) 
    + Ciso(5,0)*v1(0) + Ciso(5,2)*v3(2) - Ciso(1,5)*da2 - Ciso(2,3)*da3);
  real df32 = beta1 * (Ciso(1,0)*v2(0) + Ciso(1,1)*v3(1) + Ciso(1,3)*(v3(0) + 
    v2(1)) - Ciso(2,1)*da3);
  real df33 = beta1 * (Ciso(2,0)*v1(0) + Ciso(2,2)*v3(2) + Ciso(2,5)*(v3(0) + 
    v1(2)) - Ciso(1,2)*da2);
  real df41 = beta2 * (Ciso(0,0)*v4(0) - (Ciso(0,0) + Ciso(1,0) + Ciso(2,0))*da5);
  real df42 = beta2 * (Ciso(1,1)*v4(1) - (Ciso(0,1) + Ciso(1,1) + Ciso(2,1))*da6);
  real df43 = beta2 * (Ciso(2,2)*v4(2) - (Ciso(0,2) + Ciso(1,2) + Ciso(2,2))*da4);
  df1(0) += df11; df1(1) += df12; df1(2) += df13;
  df2(0) += df21; df2(1) += df22; df2(2) += df23;
  df3(0) += df31; df3(1) += df32; df3(2) += df33;
  df4(0) += df41; df4(1) += df42; df4(2) += df43;

  // Step 7: Calculate stabilization forces (fStab).
  afb::ColumnVector fStab(24);
  fStab.ClearAll();
  for (int i = 0; i < 4; i++)
  {
    const afb::ColumnVector& fi = absref<afb::ColumnVector>(hourglassForces_[i]);
    for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
    {
      real gamma = (*stabilizationVector[i])(nodeIdx);
      real fs1 = gamma * (J(0,0)*fi(0) + J(0,1)*fi(1) + J(0,2)*fi(2));
      real fs2 = gamma * (J(1,0)*fi(0) + J(1,1)*fi(1) + J(1,2)*fi(2));
      real fs3 = gamma * (J(2,0)*fi(0) + J(2,1)*fi(1) + J(2,2)*fi(2));

      fStab(3*nodeIdx    ) += fs1;
      fStab(3*nodeIdx + 1) += fs2;
      fStab(3*nodeIdx + 2) += fs3;
    }
  }

  // Calculate hourglass energy increment
  real hourglassEnergyIncrement = 0;
  for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
  {
    real gamma1 = (*stabilizationVector[0])(nodeIdx);
    real gamma2 = (*stabilizationVector[1])(nodeIdx);
    real gamma3 = (*stabilizationVector[2])(nodeIdx);
    real gamma4 = (*stabilizationVector[3])(nodeIdx);
    real dfs1 = gamma1 * (J(0,0)*df11 + J(0,1)*df12 + J(0,2)*df13)
              + gamma2 * (J(0,0)*df21 + J(0,1)*df22 + J(0,2)*df23)
              + gamma3 * (J(0,0)*df31 + J(0,1)*df32 + J(0,2)*df33)
              + gamma4 * (J(0,0)*df41 + J(0,1)*df42 + J(0,2)*df43);
    real dfs2 = gamma1 * (J(1,0)*df11 + J(1,1)*df12 + J(1,2)*df13)
              + gamma2 * (J(1,0)*df21 + J(1,1)*df22 + J(1,2)*df23)
              + gamma3 * (J(1,0)*df31 + J(1,1)*df32 + J(1,2)*df33)
              + gamma4 * (J(1,0)*df41 + J(1,1)*df42 + J(1,2)*df43);
    real dfs3 = gamma1 * (J(2,0)*df11 + J(2,1)*df12 + J(2,2)*df13)
              + gamma2 * (J(2,0)*df21 + J(2,1)*df22 + J(2,2)*df23)
              + gamma3 * (J(2,0)*df31 + J(2,1)*df32 + J(2,2)*df33)
              + gamma4 * (J(2,0)*df41 + J(2,1)*df42 + J(2,2)*df43);
    hourglassEnergyIncrement += dfs1*du(3*nodeIdx) + dfs2*du(3*nodeIdx + 1) + 
      dfs3*du(3*nodeIdx + 2);
  }
  hourglassEnergy_ += hourglassEnergyIncrement;

  // Step 8: Stabilize internal force
  afb::VectorSum(internalForce, 1.0, internalForce, 1.0, fStab);
}

afu::Uuid adf::LinearHexahedralPusoFormulation::GetTypeId( void ) const
{
  // 993F06D4-A4B7-4B49-B5D3-42DA6FE86AE3
  int bytes[16] = {0x99, 0x3F, 0x06, 0xD4, 0xA4, 0xB7, 0x4B, 0x49, 
                   0xB5, 0xD3, 0x42, 0xDA, 0x6F, 0xE8, 0x6A, 0xE3};
  return afu::Uuid(bytes);
}

// void hexa::StabilizeInternalForce_org( afb::Vector& internalForce, 
//                                    const afb::Vector& displacementIncrement )
// {
//   id_type id = Element().GetInternalId();
// 
//   // These are some shorthands that we are going to use
//   const afb::Matrix& J = absref<afb::Matrix>(jacobian_);
//   const afb::Matrix& B = absref<afb::Matrix>(Bmatrix_);
//   const afb::Vector& du = displacementIncrement;
//   afb::Matrix& X = absref<afb::Matrix>(nodeCoordinates_);
//   const afb::Matrix& C = Element().Material().GetMaterialTensor();
// 
//   // Step 1: Calculate stabilization vectors gamma_i (Eq. 11)
//   real stabilizationVector[3][4][8];
//   for (int dofIdx = 0; dofIdx < 3; dofIdx++)
//   {
//     for (int modeIdx = 0; modeIdx < 4; modeIdx++) // iterate through vectors
//     {
//       real *gamma = stabilizationVector[dofIdx][modeIdx];
//       const real *h = hourglassVector_[modeIdx];
//       real dotProduct = 0;
//       for (int m = 0; m < 8; m++) // dot product h_i * x_j
//       {
//         dotProduct += h[m] * X(m, dofIdx);
//       }
//       for (int k = 0; k < 8; k++) // iterate through vector indices
//       {
//         gamma[k] = 0.125 * (h[k] - dotProduct*B(dofIdx, k));
//       }
//     }
//   }
// 
//   // Step 2: Calculate hourglass displacements (already in isoparametric 
//   // domain, Eq. 8)
//   real hourglassDisplacement[4][3];
//   for (int i = 0; i < 4; i++)
//   {
//     real v_1 = 0, v_2 = 0, v_3 = 0;
//     for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
//     {
//       // in generalized coordinates...
//       v_1 += stabilizationVector[0][i][nodeIdx] * du(3*nodeIdx    );
//       v_2 += stabilizationVector[1][i][nodeIdx] * du(3*nodeIdx + 1);
//       v_3 += stabilizationVector[2][i][nodeIdx] * du(3*nodeIdx + 2);
//     }
// 
//     // ...now in isoparametric coordinates
//     for (int k = 0; k < 3; k++)
//     {
//       hourglassDisplacement[k][i] = J(0, k)*v_1 + J(1, k)*v_2 + J(2, k)*v_3;
//     }
//   }
// 
//   // Step 3: Calculate hat(J)_0^(-1), which maps strains to isoparametric 
//   // coordinates (Eq. 3.61 pg. 3.14, LS-DYNA Theory Manual)
//   afb::ColumnVector strainJacobian(6);  
//   real jSquaredNorm[3];
//   for (int i = 0; i < 3; i++)
//   {
//     real squaredNorm = J(0,i)*J(0,i) + J(1,i)*J(1,i) + J(2,i)*J(2,i);
//     jSquaredNorm[i] = squaredNorm;
//   }
//   strainJacobian(0) = 1.0 / jSquaredNorm[0];
//   strainJacobian(1) = 1.0 / jSquaredNorm[1];
//   strainJacobian(2) = 1.0 / jSquaredNorm[2];
//   strainJacobian(3) = 1.0 / sqrt(jSquaredNorm[0] * jSquaredNorm[1]);
//   strainJacobian(4) = 1.0 / sqrt(jSquaredNorm[2] * jSquaredNorm[1]);
//   strainJacobian(5) = 1.0 / sqrt(jSquaredNorm[0] * jSquaredNorm[2]);
// 
//   // Step 4: Map material stiffness matrix to isoparametric space.
//   afb::DenseMatrix Ciso(6, 6);
//   for (int i = 0; i < 6; i++)
//   {
//     for (int j = 0; j < 6; j++)
//     {
//       real c_ij = strainJacobian(i) * C(i, j) * strainJacobian(j);
//       Ciso(i, j) = c_ij;
//     }
//   }
// 
//   // Step 5: Calculate internal degrees of freedom for enhanced strain.
//   real *v1 = hourglassDisplacement[0];
//   real *v2 = hourglassDisplacement[1];
//   real *v3 = hourglassDisplacement[2];
//   real *v4 = hourglassDisplacement[3];
//   real H = Ciso(0,0) + Ciso(1,1) + Ciso(2,2) + 2.0*(Ciso(0,1) + Ciso(1,2) + Ciso(0,2));
//   real da1 = 1.0/Ciso(0,0) * (Ciso(0,1)*v1[1] + Ciso(0,4)*v2[1] + Ciso(0,2)*v2[2] + Ciso(0,4)*v1[2]);
//   real da2 = 1.0/Ciso(1,1) * (Ciso(1,0)*v1[0] + Ciso(1,5)*v3[0] + Ciso(1,2)*v3[2]);
//   real da3 = 1.0/Ciso(2,2) * (Ciso(2,0)*v2[0] + Ciso(2,3)*v3[0] + Ciso(2,1)*v3[1] + Ciso(2,3)*v2[1]);
//   real da4 = v4[2]/H * (Ciso(0,2) + Ciso(1,2) + Ciso(2,2));
//   real da5 = v4[0]/H * (Ciso(0,0) + Ciso(1,0) + Ciso(2,0));
//   real da6 = v4[1]/H * (Ciso(0,1) + Ciso(1,1) + Ciso(2,1));
// 
//   // Step 6: Calculate hourglass force increment
//   afb::Vector& df1 = absref<afb::Vector>(hourglassForces_[0]);
//   afb::Vector& df2 = absref<afb::Vector>(hourglassForces_[1]);
//   afb::Vector& df3 = absref<afb::Vector>(hourglassForces_[2]);
//   afb::Vector& df4 = absref<afb::Vector>(hourglassForces_[3]);
//   real beta1 = 8.0/3.0 * volume_;
//   real beta2 = 8.0/9.0 * volume_;
//   real df11 = beta1 * (Ciso(0,0)*v1[0] + Ciso(0,2)*v3[2] + Ciso(0,5)*(v3[0] + v1[2]) - Ciso(1,0)*da2);
//   real df12 = beta1 * (Ciso(1,1)*v1[1] + Ciso(1,2)*v2[2] + Ciso(1,4)*(v2[1] + v1[2]) - Ciso(0,1)*da1);
//   real df13 = beta1 * (Ciso(4,4)*(v2[1] + v1[2]) + Ciso(5,5)*(v3[0] + v1[2]) + Ciso(5,0)*v1[0] + Ciso(4,1)*v1[1]
//               + Ciso(4,2)*v2[2] + Ciso(5,2)*v3[2] - Ciso(0,4)*da1);
//   real df21 = beta1 * (Ciso(0,0)*v2[0] + Ciso(0,1)*v3[1] + Ciso(0,3)*(v3[0] + v2[1]) - Ciso(2,0)*da3);
//   real df22 = beta1 * (Ciso(3,3)*(v3[0] + v2[1]) + Ciso(4,4)*(v2[1] + v1[2]) + Ciso(3,0)*v2[0] + Ciso(3,1)*v3[1] 
//               + Ciso(4,1)*v1[1] + Ciso(4,2)*v2[2] - Ciso(0,4)*da1 - Ciso(2,3)*da3);
//   real df23 = beta1 * (Ciso(2,1)*v1[1] + Ciso(2,2)*v2[2] + Ciso(2,4)*(v2[1] + v1[2]) - Ciso(0,2)*da1);
//   real df31 = beta1 * (Ciso(3,3)*(v3[0] + v2[1]) + Ciso(5,5)*(v3[0] + v1[2]) + Ciso(3,0)*v2[0] + Ciso(3,1)*v3[1] 
//               + Ciso(5,0)*v1[0] + Ciso(5,2)*v3[2] - Ciso(1,5)*da2 - Ciso(2,3)*da3);
//   real df32 = beta1 * (Ciso(1,0)*v2[0] + Ciso(1,1)*v3[1] + Ciso(1,3)*(v3[0] + v2[1]) - Ciso(2,1)*da3);
//   real df33 = beta1 * (Ciso(2,0)*v1[0] + Ciso(2,2)*v3[2] + Ciso(2,5)*(v3[0] + v1[2]) - Ciso(1,2)*da2);
//   real df41 = beta2 * (Ciso(0,0)*v4[0] - (Ciso(0,0) + Ciso(1,0) + Ciso(2,0))*da5);
//   real df42 = beta2 * (Ciso(1,1)*v4[1] - (Ciso(0,1) + Ciso(1,1) + Ciso(2,1))*da6);
//   real df43 = beta2 * (Ciso(2,2)*v4[2] - (Ciso(0,2) + Ciso(1,2) + Ciso(2,2))*da4);
//   df1(0) += df11; df1(1) += df12; df1(2) += df13;
//   df2(0) += df21; df2(1) += df22; df2(2) += df23;
//   df3(0) += df31; df3(1) += df32; df3(2) += df33;
//   df4(0) += df41; df4(1) += df42; df4(2) += df43;
// 
//   // Step 7: Calculate stabilization forces (fStab).
//   afb::ColumnVector fStab(24);
//   afb::ColumnVector dFstab(24);
//   fStab.ClearAll();
//   for (int i = 0; i < 4; i++)
//   {
//     const afb::Vector& fi = absref<afb::Vector>(hourglassForces_[i]);
//     for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
//     {
//       real gammaX = stabilizationVector[0][i][nodeIdx];
//       real gammaY = stabilizationVector[1][i][nodeIdx];
//       real gammaZ = stabilizationVector[2][i][nodeIdx];
//       real fs1 = gammaX * (J(0,0)*fi(0) + J(0,1)*fi(1) + J(0,2)*fi(2));
//       real fs2 = gammaY * (J(1,0)*fi(0) + J(1,1)*fi(1) + J(1,2)*fi(2));
//       real fs3 = gammaZ * (J(2,0)*fi(0) + J(2,1)*fi(1) + J(2,2)*fi(2));
// 
//       fStab(3*nodeIdx    ) += fs1;
//       fStab(3*nodeIdx + 1) += fs2;
//       fStab(3*nodeIdx + 2) += fs3;
//     }
//   }
// 
//   // Calculate hourglass energy increment
//   real hourglassEnergyIncrement = 0;
//   for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
//   {
//     real gammaX1 = stabilizationVector[0][0][nodeIdx];
//     real gammaY1 = stabilizationVector[1][0][nodeIdx];
//     real gammaZ1 = stabilizationVector[2][0][nodeIdx];
//     real gammaX2 = stabilizationVector[0][1][nodeIdx];
//     real gammaY2 = stabilizationVector[1][1][nodeIdx];
//     real gammaZ2 = stabilizationVector[2][1][nodeIdx];
//     real gammaX3 = stabilizationVector[0][2][nodeIdx];
//     real gammaY3 = stabilizationVector[1][2][nodeIdx];
//     real gammaZ3 = stabilizationVector[2][2][nodeIdx];
//     real gammaX4 = stabilizationVector[0][3][nodeIdx];
//     real gammaY4 = stabilizationVector[1][3][nodeIdx];
//     real gammaZ4 = stabilizationVector[2][3][nodeIdx];
// 
//     real dfs1 = gammaX1 * (J(0,0)*df11 + J(0,1)*df12 + J(0,2)*df13)
//               + gammaX2 * (J(0,0)*df21 + J(0,1)*df22 + J(0,2)*df23)
//               + gammaX3 * (J(0,0)*df31 + J(0,1)*df32 + J(0,2)*df33)
//               + gammaX4 * (J(0,0)*df41 + J(0,1)*df42 + J(0,2)*df43);
//     real dfs2 = gammaY1 * (J(1,0)*df11 + J(1,1)*df12 + J(1,2)*df13)
//               + gammaY2 * (J(1,0)*df21 + J(1,1)*df22 + J(1,2)*df23)
//               + gammaY3 * (J(1,0)*df31 + J(1,1)*df32 + J(1,2)*df33)
//               + gammaY4 * (J(1,0)*df41 + J(1,1)*df42 + J(1,2)*df43);
//     real dfs3 = gammaZ1 * (J(2,0)*df11 + J(2,1)*df12 + J(2,2)*df13)
//               + gammaZ2 * (J(2,0)*df21 + J(2,1)*df22 + J(2,2)*df23)
//               + gammaZ3 * (J(2,0)*df31 + J(2,1)*df32 + J(2,2)*df33)
//               + gammaZ4 * (J(2,0)*df41 + J(2,1)*df42 + J(2,2)*df43);
// 
//     hourglassEnergyIncrement += dfs1*du(3*nodeIdx) + dfs2*du(3*nodeIdx + 1) + dfs3*du(3*nodeIdx + 2);
//   }
//   hourglassEnergy_ += hourglassEnergyIncrement;
// 
//   // Step 8: Stabilize internal force
//   afb::VectorAlgebra::Sum(internalForce, 1.0, internalForce, 1.0, fStab);
// }

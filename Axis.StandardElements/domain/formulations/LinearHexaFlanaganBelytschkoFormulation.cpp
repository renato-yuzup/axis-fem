#include <exception>
#include "LinearHexaFlanaganBelytschkoFormulation.hpp"
#include "LinearHexaFlanaganBelytschkoHelper.hpp"
#include "domain/elements/Node.hpp"
#include "domain/elements/DoF.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "foundation/NotImplementedException.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"
#include "foundation/memory/pointer.hpp"

namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adm = axis::domain::materials;
namespace adf = axis::domain::formulations;
namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;
namespace afu = axis::foundation::uuids;

namespace {
  typedef adf::LinearHexaFlanaganBelytschkoFormulation hexa_belytschko;
}

hexa_belytschko::LinearHexaFlanaganBelytschkoFormulation( 
  real antiHourglassRatio ) : Bmatrix_(NULLPTR), nodePosition_(NULLPTR), 
  hourglassForce_(NULLPTR), antiHourglassRatio_(antiHourglassRatio)
{
  massMatrix_ = NULLPTR;
  stiffnessMatrix_ = NULLPTR;
  hourglassEnergy_ = 0;
}

hexa_belytschko::~LinearHexaFlanaganBelytschkoFormulation( void )
{
  if (Bmatrix_ != NULLPTR)
  {
    absref<afb::DenseMatrix>(Bmatrix_).Destroy();
    System::ModelMemory().Deallocate(Bmatrix_);
  }
  if (nodePosition_ != NULLPTR)
  {
    absref<afb::DenseMatrix>(nodePosition_).Destroy();
    System::ModelMemory().Deallocate(nodePosition_);
  }
  if (hourglassForce_ != NULLPTR)
  {
    absref<afb::ColumnVector>(hourglassForce_).Destroy();
    System::ModelMemory().Deallocate(hourglassForce_);
  }
  if (massMatrix_ != NULLPTR) 
  {
    absref<afb::ColumnVector>(massMatrix_).Destroy();
    System::ModelMemory().Deallocate(massMatrix_);
  }
  if (stiffnessMatrix_ != NULLPTR)
  {
    absref<afb::SymmetricMatrix>(stiffnessMatrix_).Destroy();
    System::GlobalMemory().Deallocate(stiffnessMatrix_);
  }
  Bmatrix_ = NULLPTR;
  nodePosition_ = NULLPTR;
  hourglassForce_ = NULLPTR;
  massMatrix_ = NULLPTR;
  stiffnessMatrix_ = NULLPTR;
}

void hexa_belytschko::Destroy( void ) const
{
  delete this;
}

void hexa_belytschko::AllocateMemory( void )
{
  if (nodePosition_ == NULLPTR) nodePosition_ = afb::DenseMatrix::Create(8, 3);
  if (hourglassForce_ == NULLPTR) hourglassForce_ = afb::ColumnVector::Create(24);
  if (Bmatrix_ == NULLPTR) Bmatrix_ = afb::DenseMatrix::Create(3, 8);
}

void hexa_belytschko::CalculateInitialState( void )
{
  const ade::ElementGeometry& geometry = Element().Geometry();
  auto& hgForce = absref<afb::ColumnVector>(hourglassForce_);
  auto& nodeCoordinate = absref<afb::DenseMatrix>(nodePosition_);
  auto& Bmatrix = absref<afb::DenseMatrix>(Bmatrix_);
  hgForce.ClearAll();
  BuildNodeCoordinateMatrix(nodeCoordinate, geometry);
  CalculateBMatrix(Bmatrix, nodeCoordinate);
  volume_ = CalculateElementVolume(Bmatrix, nodeCoordinate);
}

void hexa_belytschko::UpdateStrain(
  const afb::ColumnVector& elementDisplacementIncrement)
{
  const auto& du = elementDisplacementIncrement; // shorthand
  const auto& B  = absref<afb::DenseMatrix>(Bmatrix_);
  auto& strain   = Element().PhysicalState().Strain();
  auto& dStrain  = Element().PhysicalState().LastStrainIncrement();

  // our B-matrix is somewhat "different", so we rearrange the matrix-vector 
  // computation normally done
  dStrain.ClearAll();
  for (int i = 0; i < 8; i++)
  {
    dStrain(0) += B(0,i)*du(3*i);                          // du1/dx
    dStrain(1) += B(1,i)*du(3*i + 1);                      // du2/dy
    dStrain(2) += B(2,i)*du(3*i + 2);                      // du3/dz
    dStrain(3) += B(2,i)*du(3*i + 1) + B(1,i)*du(3*i + 2); // du2/dz + du3/dy
    dStrain(4) += B(2,i)*du(3*i) + B(0,i)*du(3*i + 2);     // du1/dz + du3/dx
    dStrain(5) += B(1,i)*du(3*i) + B(0,i)*du(3*i + 1);     // du1/dy + du2/dx
  }
  dStrain.Scale(1.0 / volume_);
  afb::VectorSum(strain, 1.0, strain, 1.0, dStrain);
}

void hexa_belytschko::UpdateInternalForce(
  afb::ColumnVector& elementInternalForce, 
  const afb::ColumnVector& elementDisplacementIncrement, 
  const afb::ColumnVector& elementVelocity, 
  const ada::AnalysisTimeline& timeInfo )
{
  const afb::ColumnVector& stress = Element().PhysicalState().Stress();
  const real dt = timeInfo.LastTimeIncrement();

  elementInternalForce.ClearAll();
  CalculateCentroidalInternalForces(elementInternalForce, stress);
  ApplyAntiHourglassForces(elementInternalForce, elementVelocity, dt);

  // invert internal forces so that it is opposed to movement
  elementInternalForce.Scale(-1.0);
}

void hexa_belytschko::UpdateMatrices(const ade::MatrixOption& whichMatrices, 
  const afb::ColumnVector&, const afb::ColumnVector&)
{
  if (whichMatrices.DoesRequestConsistentMassMatrix())
  {
    throw axis::foundation::NotImplementedException(
      _T("This element does not implement consistent mass matrix."));
  }
  if (whichMatrices.DoesRequestStiffnessMatrix())
  {
    throw axis::foundation::NotImplementedException(
      _T("This element does not implement stiffness matrix."));
  }
  if (whichMatrices.DoesRequestLumpedMassMatrix()) CalculateLumpedMassMatrix();
}

real hexa_belytschko::GetCriticalTimestep(const afb::ColumnVector&) const
{
  throw std::exception("The method or operation is not implemented.");
}

void hexa_belytschko::ClearMemory( void )
{
  absref<afb::DenseMatrix>(Bmatrix_).Destroy();
  absref<afb::DenseMatrix>(nodePosition_).Destroy();
  System::ModelMemory().Deallocate(Bmatrix_);
  System::ModelMemory().Deallocate(nodePosition_);
  Bmatrix_ = NULLPTR;
  nodePosition_ = NULLPTR;
}

const afb::SymmetricMatrix& hexa_belytschko::GetStiffness( void ) const
{
  return absref<afb::SymmetricMatrix>(stiffnessMatrix_);
}

const afb::SymmetricMatrix& hexa_belytschko::GetConsistentMass( void ) const
{
  throw axis::foundation::NotImplementedException(
    _T("This element does not implement consistent mass matrix."));
}

const afb::ColumnVector& hexa_belytschko::GetLumpedMass( void ) const
{
  return absref<afb::ColumnVector>(massMatrix_);
}

real adf::LinearHexaFlanaganBelytschkoFormulation::GetTotalArtificialEnergy(
  void) const
{
  return hourglassEnergy_;
}

void hexa_belytschko::CalculateLumpedMassMatrix( void )
{
  adm::MaterialModel& material = Element().Material();
  real massPerNode = material.Density() * volume_ / 8.0;
  if (massMatrix_ == NULLPTR)
  {
    massMatrix_ = afb::ColumnVector::Create(24);
  }
  absref<afb::ColumnVector>(massMatrix_).SetAll(massPerNode);
}

void hexa_belytschko::CalculateCentroidalInternalForces( 
  afb::ColumnVector& internalForce, const afb::ColumnVector& stress )
{
  const afb::DenseMatrix& B = absref<afb::DenseMatrix>(Bmatrix_);
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
}

void hexa_belytschko::ApplyAntiHourglassForces( 
  afb::ColumnVector& internalForce, const afb::ColumnVector& elementVelocity,
  real timeIncrement )
{
  const afb::DenseMatrix& B = absref<afb::DenseMatrix>(Bmatrix_);
  const afb::DenseMatrix& X = absref<afb::DenseMatrix>(nodePosition_);
  const afb::ColumnVector& v = elementVelocity;
  const real dt = timeIncrement;
  const adm::MaterialModel& material = Element().Material();
  const real bulkModulus = material.GetBulkModulus();
  const real shearModulus = material.GetShearModulus();

  /************************************************************************/
  /* Calculate hourglass shape vectors (gamma, Eq.                        */
  /************************************************************************/
  real hourglassShapeVector[3][4][8];
  for (int dofIdx = 0; dofIdx < 3; dofIdx++)
  {
    for (int vecIdx = 0; vecIdx < 4; vecIdx++)
    {
      real *gamma = hourglassShapeVector[dofIdx][vecIdx];
      for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
      {
        real bPerVol = B(dofIdx, nodeIdx) / volume_;
        real h = hourglassVectors[vecIdx][nodeIdx];
        gamma[nodeIdx] = h - bPerVol*X(nodeIdx, dofIdx)*h;
      }
    }
  }

  /************************************************************************/
  /* Calculate stiffness coefficient                                      */
  /************************************************************************/
  real stiffnessCoef[3][4];
  real constantPart = antiHourglassRatio_ * dt * (bulkModulus + 
    4.0/3.0*shearModulus) / (3.0 * volume_);
  for (int dofIdx = 0; dofIdx < 3; dofIdx++)
  {
    for (int vecIdx = 0; vecIdx < 4; vecIdx++)
    {
      real dotQ = 0;
      real bb = 0;
      for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
      {
        dotQ += v(3*nodeIdx + dofIdx) * 
          hourglassShapeVector[dofIdx][vecIdx][nodeIdx];
        bb += B(dofIdx, nodeIdx) * B(dofIdx, nodeIdx);
      }
      dotQ /= sqrt(8.0);
      stiffnessCoef[dofIdx][vecIdx] = constantPart * bb * dotQ;
    }
  }

  /************************************************************************/
  /* Calculate hourglasses forces, which are linear combination of        */
  /* each hourglass shape vectors.                                        */
  /************************************************************************/
  afb::ColumnVector& hourglassForce = absref<afb::ColumnVector>(hourglassForce_);
  real hourglassWorkRatio = 0;
  for (int dofIdx = 0; dofIdx < 3; dofIdx++)
  {
    for (int vecIdx = 0; vecIdx < 4; vecIdx++)
    {
      real Q = stiffnessCoef[dofIdx][vecIdx];
      real scalar = Q / sqrt(8.0);
      for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
      {
        real dfij = scalar * hourglassShapeVector[dofIdx][vecIdx][nodeIdx];
        hourglassForce(3*nodeIdx + dofIdx) += dfij;
        hourglassWorkRatio += dfij * MATH_ABS(v(3*nodeIdx + dofIdx));
      }
    }
  }

  /************************************************************************/
  /* Update hourglass energy.                                             */
  /************************************************************************/
  real dHourglassEnergy = hourglassWorkRatio * dt;
  hourglassEnergy_ += dHourglassEnergy;

  /************************************************************************/
  /* Update element internal forces.                                      */
  /************************************************************************/
  afb::VectorSum(internalForce, 1.0, internalForce, 1.0, hourglassForce);
}

afu::Uuid adf::LinearHexaFlanaganBelytschkoFormulation::GetTypeId( void ) const
{
  // B821D32A-8FF6-49CC-8E54-BE80ACD0969E
  int bytes[16] = {0xB8, 0x21, 0xD3, 0x2A, 0x8F, 0xF6, 0x49, 0xCC, 
                   0x8E, 0x54, 0xBE, 0x80, 0xAC, 0xD0, 0x96, 0x9E};
  return afu::Uuid(bytes);
}

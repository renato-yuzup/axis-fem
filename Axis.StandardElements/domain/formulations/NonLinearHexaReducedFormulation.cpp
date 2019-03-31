#include "NonLinearHexaReducedFormulation.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/pointer.hpp"
#include "System.hpp"
#include "NonLinearHexaReducedStrategy.hpp"
#include "nlhr_gpu_data.hpp"
#include "foundation/NotSupportedException.hpp"

namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adf = axis::domain::formulations;
namespace adi = axis::domain::integration;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

class adf::NonLinearHexaReducedFormulation::FormulationData
{
public:
  afm::RelativePointer BMatrix;
  real UpdatedJacobianDeterminant;
  afm::RelativePointer InitialJacobianInverse;
  afm::RelativePointer UpdatedJacobianInverse;
  afm::RelativePointer StiffnessMatrix;
  afm::RelativePointer ConsistentMassMatrix;
  afm::RelativePointer LumpedMassMatrix;

  FormulationData(void)
  {
    // nothing to do here
  }
  void *operator new(size_t bytes, void *ptr) // just to allow construction on 
  {                                           // pre-allocated area
    return ptr;
  }
  void operator delete(void *, void *) // just to match operator new
  {
    // nothing to do here
  }
};

namespace {
  // Derivatives of shape function for one-point quadrature
  static const real dNr[8] = 
    {-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125};
  static const real dNs[8] = 
    {-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125};
  static const real dNt[8] = 
    {-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125};

// Calculates the inverse of the jacobian matrix and its determinant in
// a specified point of space using updated node coordinates.
void CalculateUpdatedJacobianInverse( afb::DenseMatrix& Jinvi, real& detJ, 
  const ade::ElementGeometry& geometry)
{
  // Calculate jacobian matrix. The coefficients below are
  // organized as follows:
  //      [ a  b  c ]
  //  J = [ d  e  f ]
  //      [ g  h  i ]
  real a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0;
  for (int j = 0; j < 8; j++)
  {
    const ade::Node& node = geometry.GetNode(j);
    a += dNr[j]*node.CurrentX();   b += dNr[j]*node.CurrentY();   
    c += dNr[j]*node.CurrentZ();
    d += dNs[j]*node.CurrentX();   e += dNs[j]*node.CurrentY();   
    f += dNs[j]*node.CurrentZ();
    g += dNt[j]*node.CurrentX();   h += dNt[j]*node.CurrentY();   
    i += dNt[j]*node.CurrentZ();
  }
  detJ = a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;
  
  Jinvi(0,0) = (e*i - f*h) / detJ;
  Jinvi(0,1) = (c*h - b*i) / detJ;
  Jinvi(0,2) = (b*f - c*e) / detJ;
  Jinvi(1,0) = (f*g - d*i) / detJ;
  Jinvi(1,1) = (a*i - c*g) / detJ;
  Jinvi(1,2) = (c*d - a*f) / detJ;
  Jinvi(2,0) = (d*h - e*g) / detJ;
  Jinvi(2,1) = (b*g - a*h) / detJ;
  Jinvi(2,2) = (a*e - b*d) / detJ;
}

void CalculateBMatrix( afb::DenseMatrix& Bi, real& detJ, 
  afb::DenseMatrix& Jinv, const ade::ElementGeometry& geometry )
{
  CalculateUpdatedJacobianInverse(Jinv, detJ, geometry);
  Bi.ClearAll();
  for (int j = 0; j < 8; j++)
  {
    // {dNx} = [J]^(-1)*{dNr}
    real dNjx = Jinv(0,0)*dNr[j] + Jinv(0,1)*dNs[j] + Jinv(0,2)*dNt[j];
    real dNjy = Jinv(1,0)*dNr[j] + Jinv(1,1)*dNs[j] + Jinv(1,2)*dNt[j];
    real dNjz = Jinv(2,0)*dNr[j] + Jinv(2,1)*dNs[j] + Jinv(2,2)*dNt[j];

    // Fill B-matrix for current node
    Bi(0,3*j) = dNjx;
                           Bi(1,3*j + 1) = dNjy;
                                                  Bi(2,3*j + 2) = dNjz;
                           Bi(3,3*j + 1) = dNjz;  Bi(3,3*j + 2) = dNjy;
    Bi(4,3*j) = dNjz;                             Bi(4,3*j + 2) = dNjx;
    Bi(5,3*j) = dNjy;      Bi(5,3*j + 1) = dNjx;
  }
}

real CalculateFaceArea( const ade::ElementGeometry& geometry, int node1, 
  int node2, int node3, int node4 )
{
  const ade::Node& n1 = geometry.GetNode(node1);
  const ade::Node& n2 = geometry.GetNode(node2);
  const ade::Node& n3 = geometry.GetNode(node3);
  const ade::Node& n4 = geometry.GetNode(node4);

  // calculate the four edges size and then one of the diagonals
  real s1 = afb::Distance3D(n1.CurrentX(), n1.CurrentY(), n1.CurrentZ(), 
    n2.CurrentX(), n2.CurrentY(), n2.CurrentZ());
  real s2 = afb::Distance3D(n2.CurrentX(), n2.CurrentY(), n2.CurrentZ(), 
    n3.CurrentX(), n3.CurrentY(), n3.CurrentZ());
  real s3 = afb::Distance3D(n3.CurrentX(), n3.CurrentY(), n3.CurrentZ(), 
    n4.CurrentX(), n4.CurrentY(), n4.CurrentZ());
  real s4 = afb::Distance3D(n4.CurrentX(), n4.CurrentY(), n4.CurrentZ(), 
    n1.CurrentX(), n1.CurrentY(), n1.CurrentZ());
  real d1 = afb::Distance3D(n1.CurrentX(), n1.CurrentY(), n1.CurrentZ(), 
    n3.CurrentX(), n3.CurrentY(), n3.CurrentZ());

  // Using the Heron's formula, we can calculate the quadrilateral area by 
  // dividing it into two triangles

  real sp1 = (s1 + s2 + d1) / 2.0;	// semi-perimeter	
  real area1 = (sp1 - s1) * (sp1 - s2) * (sp1 - d1);
  area1 = sqrt(sp1 * area1);

  real sp2 = (s3 + s4 + d1) / 2.0;	// semi-perimeter	
  real area2 = (sp2 - s3) * (sp2 - s4) * (sp2 - d1);
  area2 = sqrt(sp2 * area2);

  return area1 + area2;
}

void CalculateDeformationGradient(afb::DenseMatrix& deformationGradient,
  const ade::ElementGeometry& geometry, const afb::DenseMatrix& jacobianInverse)
{
  // short-hands  
  const auto& Jinv = jacobianInverse;
  auto&       F    = deformationGradient;
  for (int i = 0; i < 3; ++i)
  {
    // derivatives of u in respect of r,s,t (isoparametric base)
    real dui_r = 0, dui_s = 0, dui_t = 0;
    for (int j = 0; j < 8; j++)
    {
      const auto& node = geometry[j];
      real x = (i == 0)?  node.CurrentX() : 
               (i == 1)?  node.CurrentY() : 
               /*i == 2*/ node.CurrentZ() ;
      dui_r += dNr[j] * x;
      dui_s += dNs[j] * x;
      dui_t += dNt[j] * x;
    }

    // calculate F_ij
    for (int j = 0; j < 3; j++)
    {
      real Fij = dui_r*Jinv(0,j) + dui_s*Jinv(1,j) + dui_t*Jinv(2,j);
      F(i,j) = Fij;
    }
  }
}

void CalculateIncrementalDeformationGradient(
  afb::DenseMatrix& deformationGradient,
  const afb::ColumnVector& displacementIncrement, 
  const afb::DenseMatrix& jacobianInverse)
{
  // short-hands  
  const auto& Jinv = jacobianInverse;
  auto&       F    = deformationGradient;

  afb::Identity3D(deformationGradient);
  for (int i = 0; i < 3; ++i)
  {
    // derivatives of u in respect of r,s,t (isoparametric base)
    real dui_r = 0, dui_s = 0, dui_t = 0;
    for (int j = 0; j < 8; j++)
    {
      real x = displacementIncrement(3*j + i);
      dui_r += dNr[j] * x;
      dui_s += dNs[j] * x;
      dui_t += dNt[j] * x;
    }

    // calculate F_ij
    for (int j = 0; j < 3; j++)
    {
      real Fij = dui_r*Jinv(0,j) + dui_s*Jinv(1,j) + dui_t*Jinv(2,j);
      F(i,j) += Fij;
    }
  }
}

}; // namespace

adf::NonLinearHexaReducedFormulation::NonLinearHexaReducedFormulation(void)
{
  data_ = System::ModelMemory().Allocate(sizeof(FormulationData));
  new (*data_) FormulationData(); // call constructor
}

adf::NonLinearHexaReducedFormulation::~NonLinearHexaReducedFormulation(void)
{
  System::ModelMemory().Deallocate(data_);
}

void adf::NonLinearHexaReducedFormulation::Destroy( void ) const
{
  delete this;
}

bool adf::NonLinearHexaReducedFormulation::IsNonLinearFormulation( void ) const
{
  return true;
}

void adf::NonLinearHexaReducedFormulation::AllocateMemory( void )
{
  ade::ElementGeometry& g = Element().Geometry();
  FormulationData &data = absref<FormulationData>(data_);

  // allocate B-matrix and J^{-1} matrices (initial and updated)
  data.BMatrix = afb::DenseMatrix::Create(6, 24);
  data.InitialJacobianInverse = afb::DenseMatrix::Create(3, 3);
  data.UpdatedJacobianInverse = afb::DenseMatrix::Create(3, 3);

  // reset element state
  adi::IntegrationPoint&   p = g.GetIntegrationPoint(0);
  adp::InfinitesimalState& d = p.State();
  d.Reset();
}

void adf::NonLinearHexaReducedFormulation::CalculateInitialState( void )
{
  // create short-hand, so we don't exhaustively de-reference 
  // the relative pointer (due to performance issues)
  dataPtr_ = absptr<FormulationData>(data_);

  // Calculate initial B-matrix; as this will also calculate the initial
  // jacobian, copy result to corresponding matrix
  EnsureGradientMatrices();
  afb::DenseMatrix& Jinv_initial = 
    absref<afb::DenseMatrix>(dataPtr_->InitialJacobianInverse);
  const afb::DenseMatrix& Jinv_updated = 
    absref<afb::DenseMatrix>(dataPtr_->UpdatedJacobianInverse);
  Jinv_initial.CopyFrom(Jinv_updated);

  // Determine initial deformation gradient
  auto& geometry = Element().Geometry();
  auto& p = geometry.GetIntegrationPoint(0);
  auto& F0 = p.State().DeformationGradient();
  CalculateDeformationGradient(F0, geometry, Jinv_initial);
  Element().PhysicalState().DeformationGradient().CopyFrom(F0);
}

void adf::NonLinearHexaReducedFormulation::UpdateStrain( 
  const afb::ColumnVector& elementDisplacementIncrement)
{
  // obtain element characteristics
  auto& data          = *dataPtr_;
  auto& geometry      = Element().Geometry();
  auto& Jinv_initial  = absref<afb::DenseMatrix>(data.InitialJacobianInverse);
  auto& Jinv_updated  = absref<afb::DenseMatrix>(data.UpdatedJacobianInverse);
  auto& eState        = Element().PhysicalState();
  auto& edStrain      = eState.LastStrainIncrement();
  auto& eStrain       = Element().PhysicalState().Strain();
  auto& p             = geometry.GetIntegrationPoint(0);
  auto& state         = p.State();
  // update last deformation gradient
  state.LastDeformationGradient() = state.DeformationGradient();

  // calculate new deformation gradient
  CalculateDeformationGradient(state.DeformationGradient(),
    geometry, Jinv_initial);

  // Calculate new logarithmic strain state
//   afb::SymmetricMatrix B(3,3);
//   const auto& F = state.DeformationGradient();
//   Product(B, 1.0, F, afb::NotTransposed, F, afb::Transposed);
//   auto& H = B; // just to preserve mathematical notation
//   afb::SymmetricLogarithm(H, B);
//   H.Scale(0.5);
//   if ((H(0,0) == std::numeric_limits<real>::infinity() ||
//     H(0,0) == -std::numeric_limits<real>::infinity() ||
//     H(0,0) == std::numeric_limits<real>::quiet_NaN() ||
//     H(0,0) == std::numeric_limits<real>::signaling_NaN()) &&
//     Element().GetUserId() == 1)
//   {
//     const auto& F0 = state.LastDeformationGradient();
//     std::cout << "F1 = [ " << F(0,0) << "  " << F(0,1) << "  "<< F(0,2) << "]" << std::endl;
//     std::cout << "     [ " << F(1,0) << "  " << F(1,1) << "  "<< F(1,2) << "]" << std::endl;
//     std::cout << "     [ " << F(2,0) << "  " << F(2,1) << "  "<< F(2,2) << "]" << std::endl;
//     std::cout << "F0 = [ " << F0(0,0) << "  " << F0(0,1) << "  "<< F0(0,2) << "]" << std::endl;
//     std::cout << "     [ " << F0(1,0) << "  " << F0(1,1) << "  "<< F0(1,2) << "]" << std::endl;
//     std::cout << "     [ " << F0(2,0) << "  " << F0(2,1) << "  "<< F0(2,2) << "]" << std::endl;
//     std::cout.flush();
//     throw;
//   }

  // Calculate strain increment
//   auto& dEpsilon = state.LastStrainIncrement();
//   auto& epsilon = state.Strain();
//   afb::TransformSecondTensorToVoigt(dEpsilon, H);
//   dEpsilon(3) *= 2; dEpsilon(4) *= 2; dEpsilon(5) *= 2;
//   afb::VectorSum(dEpsilon, 1.0, dEpsilon, -1.0, epsilon);
// 
//   // Update strain state
//   afb::TransformSecondTensorToVoigt(epsilon, H);
//   epsilon(3) *= 2; epsilon(4) *= 2; epsilon(5) *= 2;

  // Copy integration point state to element
  eState.CopyFrom(state);

//   if (Element().GetUserId() == 1)
//   {
//     const auto& F0 = state.LastDeformationGradient();
//     std::cout << "F1 = [ " << F(0,0) << "  " << F(0,1) << "  "<< F(0,2) << "]" << std::endl;
//     std::cout << "     [ " << F(1,0) << "  " << F(1,1) << "  "<< F(1,2) << "]" << std::endl;
//     std::cout << "     [ " << F(2,0) << "  " << F(2,1) << "  "<< F(2,2) << "]" << std::endl;
//     std::cout << "F0 = [ " << F0(0,0) << "  " << F0(0,1) << "  "<< F0(0,2) << "]" << std::endl;
//     std::cout << "     [ " << F0(1,0) << "  " << F0(1,1) << "  "<< F0(1,2) << "]" << std::endl;
//     std::cout << "     [ " << F0(2,0) << "  " << F0(2,1) << "  "<< F0(2,2) << "]" << std::endl;
//   }
}

void adf::NonLinearHexaReducedFormulation::UpdateInternalForce( 
  afb::ColumnVector& internalForce, const afb::ColumnVector&, 
  const afb::ColumnVector&, const ada::AnalysisTimeline& )
{
  FormulationData& data = *dataPtr_;
  ade::ElementGeometry& g = Element().Geometry();
  afb::DenseMatrix& Bi = absref<afb::DenseMatrix>(data.BMatrix);
  adi::IntegrationPoint& p = g.GetIntegrationPoint(0);
  adp::InfinitesimalState& state = p.State();

//   const auto& ep = state.EffectivePlasticStrain();
//   if (Element().GetUserId() == 1 && ep > 0)
//   {
//     std::cout << "ep = " << ep << std::endl;
//   }

  afb::VectorProduct(internalForce, -8.0*data.UpdatedJacobianDeterminant, 
    Bi, afb::Transposed, state.Stress(), afb::NotTransposed);
}

void adf::NonLinearHexaReducedFormulation::UpdateGeometry( void )
{
  // update B-matrix with new node coordinates
  EnsureGradientMatrices();
}

void adf::NonLinearHexaReducedFormulation::UpdateMatrices( 
  const ade::MatrixOption& whichMatrices, 
  const afb::ColumnVector& elementDisplacement, 
  const afb::ColumnVector& elementVelocity )
{
  // obtain element characteristics
  FormulationData& data = *dataPtr_;
  ade::ElementGeometry& geometry = Element().Geometry();
  adm::MaterialModel& material = Element().Material();
  bool calculateStiffness = whichMatrices.DoesRequestStiffnessMatrix();
  bool calculateConsistentMass = whichMatrices.DoesRequestConsistentMassMatrix();
  bool calculateLumpedMass = whichMatrices.DoesRequestLumpedMassMatrix();
  real density = material.Density();

  // initialize auxiliary matrices and element matrices as needed
  if (calculateStiffness) 
  {
    throw axis::foundation::NotSupportedException();
  }

  if (calculateConsistentMass)
  {
    if (data.ConsistentMassMatrix == NULLPTR)
    {
      data.ConsistentMassMatrix = afb::SymmetricMatrix::Create(24, 24);
    }
    absref<afb::SymmetricMatrix>(data.ConsistentMassMatrix).ClearAll();
  }
  if (calculateLumpedMass)
  {
    if (data.LumpedMassMatrix == NULLPTR)
    {
      data.LumpedMassMatrix = afb::ColumnVector::Create(24);
    }
  }

  real detJ = data.UpdatedJacobianDeterminant;

  if (calculateConsistentMass)
  {
    auto& Mc = absref<afb::SymmetricMatrix>(data.ConsistentMassMatrix);
    // Now, calculate mass matrix; for this task, we will divide
    // the matrix into 3-by-3 submatrices such that each one will
    // have its main diagonal filled with non-zero values.
    for(int j = 0; j < 8; j++)			// navigate by row
    {
      for (int x = j; x < 8; x++)		// navigate by column (symmetric)
      {
        for (int w = 0; w < 3; w++)			// this loop needs unwinding; navigate 
        {                               // through diagonal elements
          int rowPos = 3*j + w;
          int colPos = 3*x + w;
          Mc(rowPos, colPos) = 0.125*detJ*density; // Nj(0)*Nx(0)*w_0 = 0.125
        }
      }
    }
  }

  if (calculateLumpedMass)
  {
    real massPerNode = density * detJ;
    auto& Md = absref<afb::ColumnVector>(data.LumpedMassMatrix);
    Md.SetAll(massPerNode);
  }
}

void adf::NonLinearHexaReducedFormulation::ClearMemory( void )
{
  // obtain element characteristics
  FormulationData& data = *dataPtr_;
  if (data.BMatrix != NULLPTR)
  {
    System::ModelMemory().Deallocate(data.BMatrix);
    data.BMatrix = NULLPTR;
  }
  if (data.InitialJacobianInverse != NULLPTR)
  {
    System::ModelMemory().Deallocate(data.InitialJacobianInverse);
    data.InitialJacobianInverse = NULLPTR;
  }
  if (data.StiffnessMatrix != NULLPTR)
  {
    System::ModelMemory().Deallocate(data.StiffnessMatrix);
    data.StiffnessMatrix = NULLPTR;
  }
  if (data.ConsistentMassMatrix != NULLPTR)
  {
    System::ModelMemory().Deallocate(data.ConsistentMassMatrix);
    data.ConsistentMassMatrix = NULLPTR;
  }
  if (data.LumpedMassMatrix != NULLPTR)
  {
    System::ModelMemory().Deallocate(data.LumpedMassMatrix);
    data.LumpedMassMatrix = NULLPTR;
  }
}

real adf::NonLinearHexaReducedFormulation::GetCriticalTimestep( 
  const afb::ColumnVector& modelDisplacement ) const
{
  FormulationData& data = *dataPtr_;
  const ade::ElementGeometry& geometry = Element().Geometry();
  const adm::MaterialModel& material = Element().Material();
  real characteristicLength = GetCharacteristicLength();
  return characteristicLength / material.GetWavePropagationSpeed();  
}

const afb::SymmetricMatrix& 
  adf::NonLinearHexaReducedFormulation::GetStiffness(void) const
{
  return absref<afb::SymmetricMatrix>(dataPtr_->StiffnessMatrix);
}

const afb::SymmetricMatrix& 
  adf::NonLinearHexaReducedFormulation::GetConsistentMass( void ) const
{
  return absref<afb::SymmetricMatrix>(dataPtr_->ConsistentMassMatrix);
}

const afb::ColumnVector& 
  adf::NonLinearHexaReducedFormulation::GetLumpedMass( void ) const
{
  return absref<afb::ColumnVector>(dataPtr_->LumpedMassMatrix);
}

real adf::NonLinearHexaReducedFormulation::GetTotalArtificialEnergy(void) const
{
  return 0;
}

void adf::NonLinearHexaReducedFormulation::EnsureGradientMatrices( void )
{
  FormulationData& data = *dataPtr_;
  ade::ElementGeometry& geometry = Element().Geometry();
  afb::DenseMatrix& Bi = absref<afb::DenseMatrix>(data.BMatrix);
  afb::DenseMatrix& Jinv = absref<afb::DenseMatrix>(data.UpdatedJacobianInverse);
  real& detJ = data.UpdatedJacobianDeterminant;
  CalculateBMatrix(Bi, detJ, Jinv, geometry);
}

real adf::NonLinearHexaReducedFormulation::GetCharacteristicLength( void ) const
{
  /*
	 * The characteristic length of the hexahedron is given by the 
	 * element volume divided by the area of the largest side.
	*/
  const ade::ElementGeometry& geometry = Element().Geometry();
  const FormulationData& data = *dataPtr_;

	// the initial jacobian determinant gives the initial element volume
	real elementVolume = 8 * data.UpdatedJacobianDeterminant;

	// calculate area of each face and lookup for the largest one
	real faceArea[6] = {CalculateFaceArea(geometry, 0, 1, 2, 3),
	                    CalculateFaceArea(geometry, 2, 3, 7, 6),
	                    CalculateFaceArea(geometry, 3, 0, 4, 7),
	                    CalculateFaceArea(geometry, 1, 0, 4, 5),
	                    CalculateFaceArea(geometry, 1, 2, 6, 5),
                      CalculateFaceArea(geometry, 4, 5, 6, 7)};
	real largestFaceArea = faceArea[0];
	for (int i = 1; i < 6; ++i)
	{
		if (faceArea[i] > largestFaceArea)
		{
			largestFaceArea = faceArea[i];
		}
	}

  return elementVolume / largestFaceArea;
}

afu::Uuid adf::NonLinearHexaReducedFormulation::GetTypeId( void ) const
{
  // 8595282D-CD53-40EB-B4CF-17080405DF70
  int bytes[16] = {0x85, 0x95, 0x28, 0x2D, 0xCD, 0x53, 0x40, 0xEB, 
                   0xB4, 0xCF, 0x17, 0x08, 0x04, 0x05, 0xDF, 0x70};
  return afu::Uuid(bytes);
}

bool adf::NonLinearHexaReducedFormulation::IsGPUCapable( void ) const
{
  return true;
}

size_type adf::NonLinearHexaReducedFormulation::GetGPUDataLength( void ) const
{
  return sizeof(NLHR_GPUFormulation);
}

void adf::NonLinearHexaReducedFormulation::InitializeGPUData( 
  void *baseDataAddress, real *artificialEnergy )
{
  const auto& geometry = Element().Geometry();
  NLHR_GPUFormulation &data = *(NLHR_GPUFormulation *)baseDataAddress;
  real *J0inv = data.InitialJacobianInverse;
  real *J1inv = data.UpdatedJacobianInverse;
  real &detJ = data.UpdatedJacobianDeterminant;
  real *B = data.BMatrix;
  *artificialEnergy = 0;

  // Calculate first state of jacobian and B-matrix
  real a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0;
  real a0=0, b0=0, c0=0, d0=0, e0=0, f0=0, g0=0, h0=0, i0=0;
  for (int j = 0; j < 8; j++)
  {
    const ade::Node& node = geometry.GetNode(j);
    a += dNr[j]*node.CurrentX();   b += dNr[j]*node.CurrentY();   
    c += dNr[j]*node.CurrentZ();
    d += dNs[j]*node.CurrentX();   e += dNs[j]*node.CurrentY();   
    f += dNs[j]*node.CurrentZ();
    g += dNt[j]*node.CurrentX();   h += dNt[j]*node.CurrentY();   
    i += dNt[j]*node.CurrentZ();
    a0 += dNr[j]*node.X();  b0 += dNr[j]*node.Y();  c0 += dNr[j]*node.Z();  
    d0 += dNs[j]*node.X();  e0 += dNs[j]*node.Y();  f0 += dNs[j]*node.Z();  
    g0 += dNt[j]*node.X();  h0 += dNt[j]*node.Y();  i0 += dNt[j]*node.Z();
  }
  detJ = a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;
  real detJ0 = a0*e0*i0 - a0*f0*h0 - b0*d0*i0 + b0*f0*g0 + c0*d0*h0 - c0*e0*g0;

  J0inv[0] = (e0*i0 - f0*h0) / detJ0;
  J0inv[1] = (c0*h0 - b0*i0) / detJ0;
  J0inv[2] = (b0*f0 - c0*e0) / detJ0;
  J0inv[3] = (f0*g0 - d0*i0) / detJ0;
  J0inv[4] = (a0*i0 - c0*g0) / detJ0;
  J0inv[5] = (c0*d0 - a0*f0) / detJ0;
  J0inv[6] = (d0*h0 - e0*g0) / detJ0;
  J0inv[7] = (b0*g0 - a0*h0) / detJ0;
  J0inv[8] = (a0*e0 - b0*d0) / detJ0;
  
  J1inv[0] = (e*i - f*h) / detJ;
  J1inv[1] = (c*h - b*i) / detJ;
  J1inv[2] = (b*f - c*e) / detJ;
  J1inv[3] = (f*g - d*i) / detJ;
  J1inv[4] = (a*i - c*g) / detJ;
  J1inv[5] = (c*d - a*f) / detJ;
  J1inv[6] = (d*h - e*g) / detJ;
  J1inv[7] = (b*g - a*h) / detJ;
  J1inv[8] = (a*e - b*d) / detJ;

  for (int j = 0; j < 8; j++)
  {
    // {dNx} = [J]^(-1)*{dNr}
    real dNjx = J1inv[0]*dNr[j] + J1inv[1]*dNs[j] + J1inv[2]*dNt[j];
    real dNjy = J1inv[3]*dNr[j] + J1inv[4]*dNs[j] + J1inv[5]*dNt[j];
    real dNjz = J1inv[6]*dNr[j] + J1inv[7]*dNs[j] + J1inv[8]*dNt[j];

    // Fill B-matrix for current node
    B[3*j + 0] = dNjx;
    B[3*j + 1] = dNjy;
    B[3*j + 2] = dNjz;
  }
}

adf::FormulationStrategy& 
  adf::NonLinearHexaReducedFormulation::GetGPUStrategy( void )
{
  return NonLinearHexaReducedStrategy::GetInstance();
}

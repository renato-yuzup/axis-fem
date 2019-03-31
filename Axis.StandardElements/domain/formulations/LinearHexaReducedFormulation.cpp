#include "LinearHexaReducedFormulation.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/pointer.hpp"
#include "System.hpp"
#include "LinearHexaReducedStrategy.hpp"
#include "lhr_gpu_data.hpp"

namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adf = axis::domain::formulations;
namespace adi = axis::domain::integration;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

class adf::LinearHexaReducedFormulation::FormulationData
{
public:
  afm::RelativePointer BMatrix;
  real JacobianDeterminant;
  afm::RelativePointer StiffnessMatrix;
  afm::RelativePointer ConsistentMassMatrix;
  afm::RelativePointer LumpedMassMatrix;
  bool InitializedBMatrices;

  FormulationData(void)
  {
    InitializedBMatrices = false;
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

void CalculateShapeFunctionDerivatives( real dNr[], real dNs[], real dNt[] )
{
  dNr[0] = -0.125;  dNs[0] = -0.125;  dNt[0] = -0.125;
  dNr[1] =  0.125;  dNs[1] = -0.125;  dNt[1] = -0.125;
  dNr[2] =  0.125;  dNs[2] =  0.125;  dNt[2] = -0.125;
  dNr[3] = -0.125;  dNs[3] =  0.125;  dNt[3] = -0.125;
  dNr[4] = -0.125;  dNs[4] = -0.125;  dNt[4] =  0.125;
  dNr[5] =  0.125;  dNs[5] = -0.125;  dNt[5] =  0.125;
  dNr[6] =  0.125;  dNs[6] =  0.125;  dNt[6] =  0.125;
  dNr[7] = -0.125;  dNs[7] =  0.125;  dNt[7] =  0.125;
}

// Calculates the inverse of the jacobian matrix and its determinant in
// a specified point of space.
void CalculateJacobianInverse( afb::DenseMatrix& Jinvi, real& detJ, 
  const ade::ElementGeometry& geometry, const real dNri[], const real dNsi[], 
  const real dNti[])
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
    a += dNri[j]*node.X();   b += dNri[j]*node.Y();   c += dNri[j]*node.Z();
    d += dNsi[j]*node.X();   e += dNsi[j]*node.Y();   f += dNsi[j]*node.Z();
    g += dNti[j]*node.X();   h += dNti[j]*node.Y();   i += dNti[j]*node.Z();
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
  const ade::ElementGeometry& geometry )
{
  real dNri[8], dNsi[8], dNti[8];
  afb::DenseMatrix Jinvi(3, 3);
  CalculateShapeFunctionDerivatives(dNri, dNsi, dNti);
  CalculateJacobianInverse(Jinvi, detJ, geometry, dNri, dNsi, dNti);
  Bi.ClearAll();
  for (int j = 0; j < 8; j++)
  {
    // {dNx} = [J]^(-1)*{dNr}
    real dNjx = Jinvi(0,0)*dNri[j] + Jinvi(0,1)*dNsi[j] + Jinvi(0,2)*dNti[j];
    real dNjy = Jinvi(1,0)*dNri[j] + Jinvi(1,1)*dNsi[j] + Jinvi(1,2)*dNti[j];
    real dNjz = Jinvi(2,0)*dNri[j] + Jinvi(2,1)*dNsi[j] + Jinvi(2,2)*dNti[j];

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
  real s1 = afb::Distance3D(n1.X(), n1.Y(), n1.Z(), n2.X(), n2.Y(), n2.Z());
  real s2 = afb::Distance3D(n2.X(), n2.Y(), n2.Z(), n3.X(), n3.Y(), n3.Z());
  real s3 = afb::Distance3D(n3.X(), n3.Y(), n3.Z(), n4.X(), n4.Y(), n4.Z());
  real s4 = afb::Distance3D(n4.X(), n4.Y(), n4.Z(), n1.X(), n1.Y(), n1.Z());
  real d1 = afb::Distance3D(n1.X(), n1.Y(), n1.Z(), n3.X(), n3.Y(), n3.Z());

  // Using the Heron's formula, we can calculate the quadrilateral area by dividing it
  // into two triangles

  real sp1 = (s1 + s2 + d1) / 2.0;	// semi-perimeter	
  real area1 = (sp1 - s1) * (sp1 - s2) * (sp1 - d1);
  area1 = sqrt(sp1 * area1);

  real sp2 = (s3 + s4 + d1) / 2.0;	// semi-perimeter	
  real area2 = (sp2 - s3) * (sp2 - s4) * (sp2 - d1);
  area2 = sqrt(sp2 * area2);

  return area1 + area2;
}

}; // namespace


adf::LinearHexaReducedFormulation::LinearHexaReducedFormulation(void)
{
  data_ = System::ModelMemory().Allocate(sizeof(FormulationData));
  new (*data_) FormulationData(); // call constructor
}

adf::LinearHexaReducedFormulation::~LinearHexaReducedFormulation(void)
{
  System::ModelMemory().Deallocate(data_);
}

void adf::LinearHexaReducedFormulation::Destroy( void ) const
{
  delete this;
}

void adf::LinearHexaReducedFormulation::AllocateMemory( void )
{
  ade::ElementGeometry& g = Element().Geometry();
  FormulationData &data = absref<FormulationData>(data_);

  // allocate B-matrix
  data.BMatrix = afb::DenseMatrix::Create(6, 24);

  // reset element state
  adi::IntegrationPoint& p = g.GetIntegrationPoint(0);
  adp::InfinitesimalState& d = p.State();
  d.Reset();
}

void adf::LinearHexaReducedFormulation::CalculateInitialState( void )
{
  // create short-hand, so we don't exhaustively de-reference 
  // the relative pointer (due to performance issues)
  dataPtr_ = absptr<FormulationData>(data_);
}

void adf::LinearHexaReducedFormulation::UpdateStrain( 
  const afb::ColumnVector& elementDisplacementIncrement)
{
  // obtain element characteristics
  auto& du = elementDisplacementIncrement;
  FormulationData& data = *dataPtr_;
  ade::ElementGeometry& geometry = Element().Geometry();
  afb::DenseMatrix& Bi = absref<afb::DenseMatrix>(data.BMatrix);
  afb::ColumnVector& edStrain = Element().PhysicalState().LastStrainIncrement();
  afb::ColumnVector& eStrain = Element().PhysicalState().Strain();
  adi::IntegrationPoint &p = geometry.GetIntegrationPoint(0);
  adp::InfinitesimalState& state = p.State();
  auto& strain = state.Strain();
  auto& dStrain = state.LastStrainIncrement();

  // calculate strain at the integration point
  edStrain.ClearAll();
  EnsureGradientMatrices();
  afb::VectorProduct(dStrain, 1.0, Bi, du);
  afb::VectorSum(strain, 1.0, strain, 1.0, dStrain);

  // update element strain
  edStrain.CopyFrom(dStrain);
  afb::VectorSum(eStrain, 1.0, eStrain, 1.0, edStrain);
}

void adf::LinearHexaReducedFormulation::UpdateInternalForce( 
  afb::ColumnVector& internalForce, 
  const afb::ColumnVector& elementDisplacementIncrement, 
  const afb::ColumnVector& elementVelocity, 
  const ada::AnalysisTimeline& timeInfo )
{
  FormulationData& data = *dataPtr_;
  ade::ElementGeometry& g = Element().Geometry();
  afb::DenseMatrix& Bi = absref<afb::DenseMatrix>(data.BMatrix);
  adi::IntegrationPoint& p = g.GetIntegrationPoint(0);
  adp::InfinitesimalState& state = p.State();

  EnsureGradientMatrices();
  afb::VectorProduct(internalForce, -8.0*data.JacobianDeterminant, 
                     Bi, afb::Transposed, state.Stress(), afb::NotTransposed);
}

void adf::LinearHexaReducedFormulation::UpdateMatrices( 
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
  EnsureGradientMatrices();

  // initialize auxiliary matrices and element matrices as needed
  if (calculateStiffness) 
  {
    if (data.StiffnessMatrix == NULLPTR)
    {
      data.StiffnessMatrix = afb::SymmetricMatrix::Create(24, 24);
    }
    absref<afb::SymmetricMatrix>(data.StiffnessMatrix).ClearAll();
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

  const auto& Bi = absref<afb::DenseMatrix>(data.BMatrix);
  real detJ = data.JacobianDeterminant;

  if (calculateStiffness)
  {
    const afb::DenseMatrix& D = material.GetMaterialTensor();
    auto& K = absref<afb::SymmetricMatrix>(data.StiffnessMatrix);
    // Calculate integral numerically:  
    // trans(Bxyz)*D*Bxyz * det(J) * Gauss_point_weights
    afb::DenseMatrix aux(24, 6);
    afb::Product(aux, 1.0, Bi, afb::Transposed, D, afb::NotTransposed);
    afb::AccumulateProduct(K, 8.0*detJ, aux, Bi);
  }

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
        for (int w = 0; w < 3; w++)			// this loop needs unwinding; navigate through diagonal elements
        {
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

void adf::LinearHexaReducedFormulation::ClearMemory( void )
{
  // obtain element characteristics
  FormulationData& data = *dataPtr_;
  if (data.BMatrix != NULLPTR)
  {
    System::ModelMemory().Deallocate(data.BMatrix);
    data.BMatrix = NULLPTR;
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

real adf::LinearHexaReducedFormulation::GetCriticalTimestep( 
  const afb::ColumnVector& modelDisplacement ) const
{
  FormulationData& data = *dataPtr_;
  const ade::ElementGeometry& geometry = Element().Geometry();
  const adm::MaterialModel& material = Element().Material();

  if (!data.InitializedBMatrices)
  {
    auto& Bi = absref<afb::DenseMatrix>(data.BMatrix);
    real& detJ = data.JacobianDeterminant;
    CalculateBMatrix(Bi, detJ, geometry);
  }
  data.InitializedBMatrices = true;

  real characteristicLength = GetCharacteristicLength();
  return characteristicLength / material.GetWavePropagationSpeed();  
}

const afb::SymmetricMatrix& adf::LinearHexaReducedFormulation::GetStiffness( 
  void ) const
{
  return absref<afb::SymmetricMatrix>(dataPtr_->StiffnessMatrix);
}

const afb::SymmetricMatrix& adf::LinearHexaReducedFormulation::GetConsistentMass( 
  void ) const
{
  return absref<afb::SymmetricMatrix>(dataPtr_->ConsistentMassMatrix);
}

const afb::ColumnVector& adf::LinearHexaReducedFormulation::GetLumpedMass( 
  void ) const
{
  return absref<afb::ColumnVector>(dataPtr_->LumpedMassMatrix);
}

real adf::LinearHexaReducedFormulation::GetTotalArtificialEnergy( void ) const
{
  return 0;
}

afu::Uuid adf::LinearHexaReducedFormulation::GetTypeId( void ) const
{
  // 6E8A3289-0273-49CF-8593-6B9B9AD28BE4
  int bytes[16] = {0x6E, 0x8A, 0x32, 0x89, 0x02, 0x73, 0x49, 0xCF, 
                   0x85, 0x93, 0x6B, 0x9B, 0x9A, 0xD2, 0x8B, 0xE4};
  return afu::Uuid(bytes);
}

void adf::LinearHexaReducedFormulation::EnsureGradientMatrices( void )
{
  FormulationData& data = *dataPtr_;
  ade::ElementGeometry& geometry = Element().Geometry();
  if (!data.InitializedBMatrices)
  {
    afb::DenseMatrix& Bi = absref<afb::DenseMatrix>(data.BMatrix);
    real& detJ = data.JacobianDeterminant;
    CalculateBMatrix(Bi, detJ, geometry);
  }
  data.InitializedBMatrices = true;
}

real adf::LinearHexaReducedFormulation::GetCharacteristicLength( void ) const
{
  /*
	 * The characteristic length of the linear hexahedron is given by the 
	 * element volume divided by the area of the largest side.
	*/
  const ade::ElementGeometry& geometry = Element().Geometry();
  const FormulationData& data = *dataPtr_;

	// calculate total jacobian determinant; it will say the ratio of volume change from
	// master configuration
	real elementVolume = 0;
  for (int i = 0; i < 8; i++)	
	{
		elementVolume += data.JacobianDeterminant;
	}

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

bool adf::LinearHexaReducedFormulation::IsGPUCapable( void ) const
{
  return true;
}

size_type adf::LinearHexaReducedFormulation::GetGPUDataLength( void ) const
{
  return sizeof(LHR_GPUFormulation);
}

void adf::LinearHexaReducedFormulation::InitializeGPUData( 
  void *baseDataAddress, real *artificialEnergy )
{
  *artificialEnergy = 0;
  LHR_GPUFormulation &data = *(LHR_GPUFormulation *)baseDataAddress;
  for (int i = 0; i < 144; i++)
  {
    data.BMatrix[i] = 0;
  }
  data.InitializedBMatrices = false;
  data.JacobianDeterminant = 0;
}

adf::FormulationStrategy& 
  adf::LinearHexaReducedFormulation::GetGPUStrategy( void )
{
  return LinearHexaReducedStrategy::GetInstance();
}

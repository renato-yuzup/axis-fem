#include "NonLinearHexaFlanaganBelytschkoFormulation.hpp"
#include "NonLinearHexaFlanaganBelytschkoHelper.hpp"
#include "NonLinearHexaFlanaganBelytschkoHelper_GPU.hpp"
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
#include "NonLinearHexaFBReducedStrategy.hpp"
#include "nlhr_fb_gpu_data.hpp"

namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adi = axis::domain::integration;
namespace adm = axis::domain::materials;
namespace adf = axis::domain::formulations;
namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

class adf::NonLinearHexaFlanaganBelytschkoFormulation::FormulationData
{
public:
	afm::RelativePointer BMatrix;
	afm::RelativePointer InitialJacobianInverse;
	afm::RelativePointer StiffnessMatrix;
	afm::RelativePointer ConsistentMassMatrix;
	afm::RelativePointer LumpedMassMatrix;
	afm::RelativePointer NodePosition;
	real HourglassForces[24];

	real Volume;
	real AntiHourglassRatio;
	real HourglassEnergy;

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
	typedef adf::NonLinearHexaFlanaganBelytschkoFormulation hexa_belytschko;

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




hexa_belytschko::NonLinearHexaFlanaganBelytschkoFormulation( 
	real antiHourglassRatio ) 	
{
	data_ = System::ModelMemory().Allocate(sizeof(FormulationData));
	new (*data_) FormulationData(); // call constructor
	FormulationData& data = absref<FormulationData>(data_);
	data.AntiHourglassRatio = antiHourglassRatio;
	data.HourglassEnergy = 0;
	for (int i = 0; i < 24; i++)
	{
		data.HourglassForces[i] = 0;
	}

  // create short-hand, so we don't exhaustively de-reference 
  // the relative pointer (due to performance issues)
  dataPtr_ = absptr<FormulationData>(data_);
}

hexa_belytschko::~NonLinearHexaFlanaganBelytschkoFormulation( void )
{
	System::ModelMemory().Deallocate(data_);
}

void hexa_belytschko::Destroy( void ) const
{
	delete this;
}

bool hexa_belytschko::IsNonLinearFormulation( void ) const
{
	return true;
}

void hexa_belytschko::AllocateMemory( void )
{
	ade::ElementGeometry& g = Element().Geometry();
	FormulationData &data = absref<FormulationData>(data_);

	// allocate B-matrix and J^{-1} matrices (initial and updated)
	data.BMatrix = afb::DenseMatrix::Create(3, 8);
	data.InitialJacobianInverse = afb::DenseMatrix::Create(3, 3);

	// allocate hourglass matrices
	data.NodePosition = afb::DenseMatrix::Create(8, 3);
}

void hexa_belytschko::CalculateInitialState( void )
{
	// create short-hand, so we don't exhaustively de-reference 
	// the relative pointer (due to performance issues)
	dataPtr_ = absptr<FormulationData>(data_);

	// Calculate initial B-matrix
	EnsureGradientMatrices();

  // Calculate initial jacobian matrix
	const auto& geometry = Element().Geometry();
	afb::DenseMatrix& Jinv_initial = 
		absref<afb::DenseMatrix>(dataPtr_->InitialJacobianInverse);
  real detJ;
  CalculateUpdatedJacobianInverse(Jinv_initial, detJ, geometry);

	// Determine initial deformation gradient
  auto& eState = Element().PhysicalState();
  auto& F0 = eState.DeformationGradient();
	CalculateDeformationGradient(F0, geometry, Jinv_initial);
}

void hexa_belytschko::UpdateStrain(
	const afb::ColumnVector& elementDisplacementIncrement)
{
	// obtain element characteristics
	auto& data          = *dataPtr_;
	auto& geometry      = Element().Geometry();
	auto& Jinv_initial  = absref<afb::DenseMatrix>(data.InitialJacobianInverse);
	auto& eState        = Element().PhysicalState();
	// update last deformation gradient
	eState.LastDeformationGradient() = eState.DeformationGradient();

	// calculate new deformation gradient
	CalculateDeformationGradient(eState.DeformationGradient(),
		geometry, Jinv_initial);
}

void hexa_belytschko::UpdateInternalForce(
	afb::ColumnVector& elementInternalForce, 
	const afb::ColumnVector& elementDisplacementIncrement, 
	const afb::ColumnVector& elementVelocity, 
	const ada::AnalysisTimeline& timeInfo )
{
	const afb::ColumnVector& stress = Element().PhysicalState().Stress();
	const real dt = timeInfo.NextTimeIncrement();

	elementInternalForce.ClearAll();
	CalculateCentroidalInternalForces(elementInternalForce, stress);
	ApplyAntiHourglassForces(elementInternalForce, elementVelocity, dt);

	// invert internal forces so that it is opposed to movement
	elementInternalForce.Scale(-1.0);
}

void hexa_belytschko::UpdateGeometry( void )
{
	// update B-matrix with new node coordinates
	EnsureGradientMatrices();
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
	FormulationData& data = *dataPtr_;
	const ade::ElementGeometry& geometry = Element().Geometry();
	const adm::MaterialModel& material = Element().Material();
	real characteristicLength = GetCharacteristicLength();
	return characteristicLength / material.GetWavePropagationSpeed();
}

real hexa_belytschko::GetCharacteristicLength( void ) const
{
  /*
	 * The characteristic length of the hexahedron is given by the 
	 * element volume divided by the area of the largest side.
	*/
  const ade::ElementGeometry& geometry = Element().Geometry();
  const FormulationData& data = *dataPtr_;

	// the initial jacobian determinant gives the initial element volume
	real elementVolume = data.Volume;

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

void hexa_belytschko::ClearMemory( void )
{
	absref<afb::DenseMatrix>(dataPtr_->BMatrix).Destroy();
	absref<afb::DenseMatrix>(dataPtr_->InitialJacobianInverse).Destroy();
	absref<afb::DenseMatrix>(dataPtr_->NodePosition).Destroy();
	System::ModelMemory().Deallocate(dataPtr_->BMatrix);
	System::ModelMemory().Deallocate(dataPtr_->InitialJacobianInverse);
	System::ModelMemory().Deallocate(dataPtr_->NodePosition);
}

const afb::SymmetricMatrix& hexa_belytschko::GetStiffness( void ) const
{
	return absref<afb::SymmetricMatrix>(dataPtr_->StiffnessMatrix);
}

const afb::SymmetricMatrix& hexa_belytschko::GetConsistentMass( void ) const
{
	throw axis::foundation::NotImplementedException(
		_T("This element does not implement consistent mass matrix."));
}

const afb::ColumnVector& hexa_belytschko::GetLumpedMass( void ) const
{
	return absref<afb::ColumnVector>(dataPtr_->LumpedMassMatrix);
}

real adf::NonLinearHexaFlanaganBelytschkoFormulation::GetTotalArtificialEnergy(
	void) const
{
	return dataPtr_->HourglassEnergy;
}

void hexa_belytschko::CalculateLumpedMassMatrix( void )
{
	auto& massMatrix = dataPtr_->LumpedMassMatrix;
	adm::MaterialModel& material = Element().Material();
	real massPerNode = material.Density() * dataPtr_->Volume / 8.0;
	if (massMatrix == NULLPTR)
	{
		massMatrix = afb::ColumnVector::Create(24);
	}
	absref<afb::ColumnVector>(massMatrix).SetAll(massPerNode);
	dataPtr_->LumpedMassMatrix = massMatrix;
}

void hexa_belytschko::CalculateCentroidalInternalForces( 
	afb::ColumnVector& internalForce, const afb::ColumnVector& stress )
{
	const afb::DenseMatrix& B = absref<afb::DenseMatrix>(dataPtr_->BMatrix);
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
	const afb::DenseMatrix& B = absref<afb::DenseMatrix>(dataPtr_->BMatrix);
	const afb::DenseMatrix& X = absref<afb::DenseMatrix>(dataPtr_->NodePosition);
	const real volume = dataPtr_->Volume;
	const real antiHourglassRatio = dataPtr_->AntiHourglassRatio;
	real *hourglassForce = dataPtr_->HourglassForces;
	real& hourglassEnergy = dataPtr_->HourglassEnergy;
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
				real bPerVol = B(dofIdx, nodeIdx) / volume;
				real h = hourglassVectors[vecIdx][nodeIdx];
				gamma[nodeIdx] = h - bPerVol*X(nodeIdx, dofIdx)*h;
			}
		}
	}

	/************************************************************************/
	/* Calculate stiffness coefficient                                      */
	/************************************************************************/
	real stiffnessCoef[3][4];
	real constantPart = antiHourglassRatio * dt * (bulkModulus + 
		4.0/3.0*shearModulus) / (3.0 * volume);
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
				hourglassForce[3*nodeIdx + dofIdx] += dfij;
				hourglassWorkRatio += dfij * MATH_ABS(v(3*nodeIdx + dofIdx));
			}
		}
	}

	/************************************************************************/
	/* Update hourglass energy.                                             */
	/************************************************************************/
	real dHourglassEnergy = hourglassWorkRatio * dt;
	hourglassEnergy += dHourglassEnergy;

	/************************************************************************/
	/* Update element internal forces.                                      */
	/************************************************************************/
	for (int i = 0; i < 24; i++)
	{
		internalForce(i) += hourglassForce[i];
	}
}

afu::Uuid adf::NonLinearHexaFlanaganBelytschkoFormulation::GetTypeId( void ) const
{
	// B79A5383-9374-461D-B755-A334FFC84F4B
	int bytes[16] = {
		0xB7, 0x9A, 0x53, 0x83, 0x93, 0x74, 0x46, 0x1D, 
		0xB7, 0x55, 0xA3, 0x34, 0xFF, 0xC8, 0x4F, 0x4B};
	return afu::Uuid(bytes);
}

void hexa_belytschko::EnsureGradientMatrices( void )
{
	FormulationData& data = *dataPtr_;
	ade::ElementGeometry& geometry = Element().Geometry();
	afb::DenseMatrix& Bi = absref<afb::DenseMatrix>(data.BMatrix);
	afb::DenseMatrix& Jinv = absref<afb::DenseMatrix>(data.InitialJacobianInverse);

	// Calculate B-matrix using FB-centroidal formulation
	auto& nodeCoordinate = absref<afb::DenseMatrix>(dataPtr_->NodePosition);
	auto& Bmatrix = absref<afb::DenseMatrix>(dataPtr_->BMatrix);
	BuildNodeCoordinateMatrix(nodeCoordinate, geometry);
	CalculateBMatrix(Bmatrix, nodeCoordinate);
	dataPtr_->Volume = CalculateElementVolume(Bmatrix, nodeCoordinate);
}

bool hexa_belytschko::IsGPUCapable( void ) const
{
	return true;
}

size_type hexa_belytschko::GetGPUDataLength( void ) const
{
	return sizeof(NLHRFB_GPUFormulation);
}

void hexa_belytschko::InitializeGPUData( 
	void *baseDataAddress, real *artificialEnergy )
{
	const auto& geometry = Element().Geometry();
	NLHRFB_GPUFormulation &data = *(NLHRFB_GPUFormulation *)baseDataAddress;
	real *J0inv = data.InitialJacobianInverse;
	real *B = data.BMatrix;
	*artificialEnergy = 0;
  
	// hourglass control specific
	real *nodeCoord = data.NodePosition;
	real &volume = data.Volume;
	real *hourglassForces = data.HourglassForces;
	real& antiHourglassRatio = data.AntiHourglassRatio;

	// Calculate first state of jacobian 
	real a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0;
	real a0=0, b0=0, c0=0, d0=0, e0=0, f0=0, g0=0, h0=0, i0=0;
	for (int j = 0; j < 8; j++)
	{
		const ade::Node& node = geometry.GetNode(j);
// 		a += dNr[j]*node.CurrentX();   b += dNr[j]*node.CurrentY();   
// 		c += dNr[j]*node.CurrentZ();
// 		d += dNs[j]*node.CurrentX();   e += dNs[j]*node.CurrentY();   
// 		f += dNs[j]*node.CurrentZ();
// 		g += dNt[j]*node.CurrentX();   h += dNt[j]*node.CurrentY();   
// 		i += dNt[j]*node.CurrentZ();
		a0 += dNr[j]*node.X();  b0 += dNr[j]*node.Y();  c0 += dNr[j]*node.Z();  
		d0 += dNs[j]*node.X();  e0 += dNs[j]*node.Y();  f0 += dNs[j]*node.Z();  
		g0 += dNt[j]*node.X();  h0 += dNt[j]*node.Y();  i0 += dNt[j]*node.Z();
	}
// 	detJ = a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;
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
// 
// 	J1inv[0] = (e*i - f*h) / detJ;
// 	J1inv[1] = (c*h - b*i) / detJ;
// 	J1inv[2] = (b*f - c*e) / detJ;
// 	J1inv[3] = (f*g - d*i) / detJ;
// 	J1inv[4] = (a*i - c*g) / detJ;
// 	J1inv[5] = (c*d - a*f) / detJ;
// 	J1inv[6] = (d*h - e*g) / detJ;
// 	J1inv[7] = (b*g - a*h) / detJ;
// 	J1inv[8] = (a*e - b*d) / detJ;

	// Calculate B-matrix, node coordinate matrix and element volume
	BuildNodeCoordinateMatrixForGPU(nodeCoord, geometry);
	CalculateBMatrixForGPU(B, nodeCoord);
	volume = CalculateElementVolumeForGPU(B, nodeCoord);

	// initialize hourglass values
	for (int i = 0; i < 24; ++i)
	{
		hourglassForces[i] = 0;
	}
	antiHourglassRatio = dataPtr_->AntiHourglassRatio;
}

adf::FormulationStrategy& hexa_belytschko::GetGPUStrategy( void )
{
	return NonLinearHexaFBReducedStrategy::GetInstance();
}

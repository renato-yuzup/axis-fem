#include "gpu_nlhrfb_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "../nlhr_fb_gpu_data.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/domain/elements/ElementData.hpp"
#include "yuzu/domain/elements/ElementGeometry.hpp"
#include "yuzu/domain/integration/IntegrationPoint.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

#define XN(x) nodeCoord[3*MakeIdx(nodeIdx, x) + 0]
#define YN(x) nodeCoord[3*MakeIdx(nodeIdx, x) + 1]
#define ZN(x) nodeCoord[3*MakeIdx(nodeIdx, x) + 2]

namespace ay   = axis::yuzu;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace aydi = axis::yuzu::domain::integration;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

// Derivatives of shape function for one-point quadrature
GPU_ONLY const real dNrfb[8] = 
{-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125};
GPU_ONLY const real dNsfb[8] = 
{-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125};
GPU_ONLY const real dNtfb[8] = 
{-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125};

GPU_ONLY const real hourglassVectors[4][8] = 
{{ 1,  1, -1, -1, -1, -1, 1,  1},
{ 1, -1, -1,  1, -1,  1, 1, -1},
{ 1, -1,  1, -1,  1, -1, 1, -1},
{-1,  1, -1,  1,  1, -1, 1, -1}};

// 
// GPU_ONLY
// void FBCalculateUpdatedJacobianInverse( real *Jinv, real& detJ, 
//   const ayde::ElementGeometry& geometry)
// {
//   // Calculate jacobian matrix. The coefficients below are
//   // organized as follows:
//   //      [ a  b  c ]
//   //  J = [ d  e  f ]
//   //      [ g  h  i ]
//   real a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0;
//   for (int j = 0; j < 8; j++)
//   {
//     const ayde::Node& node = geometry.GetNode(j);
//     a += dNrfb[j]*node.CurrentX();   b += dNrfb[j]*node.CurrentY();   
//     c += dNrfb[j]*node.CurrentZ();
//     d += dNsfb[j]*node.CurrentX();   e += dNsfb[j]*node.CurrentY();   
//     f += dNsfb[j]*node.CurrentZ();
//     g += dNtfb[j]*node.CurrentX();   h += dNtfb[j]*node.CurrentY();   
//     i += dNtfb[j]*node.CurrentZ();
//   }
//   detJ = a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;
//   Jinv[0] = (e*i - f*h) / detJ;
//   Jinv[1] = (c*h - b*i) / detJ;
//   Jinv[2] = (b*f - c*e) / detJ;
//   Jinv[3] = (f*g - d*i) / detJ;
//   Jinv[4] = (a*i - c*g) / detJ;
//   Jinv[5] = (c*d - a*f) / detJ;
//   Jinv[6] = (d*h - e*g) / detJ;
//   Jinv[7] = (b*g - a*h) / detJ;
//   Jinv[8] = (a*e - b*d) / detJ;
// }


GPU_ONLY inline int MakeIdx(int baseIdx, int offset)
{
	if (baseIdx < 4)
		return (offset < 5) ? (baseIdx + offset-1) % 4 :
		4 + ((baseIdx + offset-5)) % 4;
	else
		return (offset < 5) ? 4 + (baseIdx - offset + 5) % 4 :
		(baseIdx - offset + 5) % 4;
}


GPU_ONLY
void FBCalculateDeformationGradient(ayfb::DenseMatrix& deformationGradient,
  const ayde::ElementGeometry& geometry, const real *jacobianInverse)
{
  // short-hands  
  const real *Jinv    = jacobianInverse;
  ayfb::DenseMatrix &F = deformationGradient;
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
      dui_r += dNrfb[j] * x;
      dui_s += dNsfb[j] * x;
      dui_t += dNtfb[j] * x;
    }

    // calculate F_ij
    for (int j = 0; j < 3; j++)
    {
      real Fij = dui_r*Jinv[j] + dui_s*Jinv[3 + j] + dui_t*Jinv[6 + j];
      F(i,j) = Fij;
    }
  }
}


__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  UpdateNonLinearStrainFBKernel( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  uint64 elementBlockSize, ayfm::RelativePointer reducedModelPtr, 
  real currentTime, real lastTimeIncrement, real nextTimeIncrement )
{
  uint64 threadIndex = 
    ay::GetBaseThreadIndex(gridDim, blockIdx, blockDim, threadIdx);
  if (!ay::IsActiveThread(threadIndex + startIndex, numThreadsToUse)) return;
  ayde::ElementData eData(baseMemoryAddressOnGPU, threadIndex, 
    elementBlockSize);
  uint64 elementId = eData.GetId();
  ayda::ReducedNumericalModel& model = 
    axis::yabsref<ayda::ReducedNumericalModel>(reducedModelPtr);
  ayde::FiniteElement& e = model.GetElement((size_type)elementId);
  ayde::ElementGeometry& geometry = e.Geometry();

  // obtain element characteristics
  NLHRFB_GPUFormulation& data = *(NLHRFB_GPUFormulation *)eData.GetFormulationBlock();
  const real *J0inv = data.InitialJacobianInverse;
  aydp::InfinitesimalState& state = e.PhysicalState();

  // Update last deformation gradient
  state.LastDeformationGradient() = state.DeformationGradient();

  // Calculate new deformation gradient
  FBCalculateDeformationGradient(state.DeformationGradient(), geometry, J0inv);
}


__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  UpdateNonLinearInternalForceFBKernel( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  uint64 elementBlockSize, ayfm::RelativePointer reducedModelPtr, 
  real currentTime, real lastTimeIncrement, real nextTimeIncrement )
{
  uint64 threadIndex = 
    ay::GetBaseThreadIndex(gridDim, blockIdx, blockDim, threadIdx);
  if (!ay::IsActiveThread(threadIndex + startIndex, numThreadsToUse)) return;
  ayde::ElementData eData(baseMemoryAddressOnGPU, threadIndex, 
    elementBlockSize);
  uint64 elementId = eData.GetId();
  ayda::ReducedNumericalModel& model = 
    axis::yabsref<ayda::ReducedNumericalModel>(reducedModelPtr);
  ayde::FiniteElement& e = model.GetElement((size_type)elementId);
  ayde::ElementGeometry& geometry = e.Geometry();

  // obtain element characteristics
  NLHRFB_GPUFormulation& data = *(NLHRFB_GPUFormulation *)eData.GetFormulationBlock();
  const real *B = data.BMatrix;
  aydp::InfinitesimalState& state = e.PhysicalState();
  const ayfb::ColumnVector &stress = state.Stress();
  real *internalForce = eData.GetOutputBuffer();
  const real dt = nextTimeIncrement;

	/************************************************************************/
	/* 1) Calculate centroidal forces                                       */
	/************************************************************************/
	for (int i = 0; i < 8; i++)
	{
		real f1 = B[0  + i]*stress(0) + B[16 + i]*stress(4) + B[8 + i]*stress(5);
		real f2 = B[8  + i]*stress(1) + B[16 + i]*stress(3) + B[0 + i]*stress(5);
		real f3 = B[16 + i]*stress(2) + B[8  + i]*stress(3) + B[0 + i]*stress(4);

		internalForce[3*i  ] = -f1;
		internalForce[3*i+1] = -f2;
		internalForce[3*i+2] = -f3;
	}

	/************************************************************************/
	/* 2) Calculate anti-hourglass forces                                   */
	/************************************************************************/
  const real *X = data.NodePosition;
  real *hourglassForce = data.HourglassForces;
  const real volume = data.Volume;
  const real antiHourglassRatio = data.AntiHourglassRatio;
	real& hourglassEnergy = eData.ArtificialEnergy();
	real v[24]; // element velocity
	const real bulkModulus = eData.BulkModulus();
	const real shearModulus = eData.ShearModulus();
 
  // Obtain element velocity
  ayfb::ColumnVector& globalVeloc = model.Kinematics().Velocity();
  for (int i = 0; i < 8; i++)
  {
    const ayde::Node& node = geometry[i];
    id_type nodeId = node.GetInternalId();
    v[3*i + 0] = globalVeloc(3*nodeId + 0);
    v[3*i + 1] = globalVeloc(3*nodeId + 1);
    v[3*i + 2] = globalVeloc(3*nodeId + 2);
  }

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
				real bPerVol = B[8*dofIdx + nodeIdx] / volume;
				real h = hourglassVectors[vecIdx][nodeIdx];
				gamma[nodeIdx] = h - bPerVol*X[3*nodeIdx + dofIdx]*h;
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
				dotQ += v[3*nodeIdx + dofIdx] * 
					hourglassShapeVector[dofIdx][vecIdx][nodeIdx];
				bb += B[8*dofIdx + nodeIdx] * B[8*dofIdx + nodeIdx];
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
				hourglassWorkRatio += dfij * abs(v[3*nodeIdx + dofIdx]);
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
		internalForce[i] -= hourglassForce[i];
	}

}



__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  UpdateNonLinearGeometryFBKernel( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  uint64 elementBlockSize, ayfm::RelativePointer reducedModelPtr, 
  real currentTime, real lastTimeIncrement, real nextTimeIncrement )
{
  uint64 threadIndex = 
    ay::GetBaseThreadIndex(gridDim, blockIdx, blockDim, threadIdx);
  if (!ay::IsActiveThread(threadIndex + startIndex, numThreadsToUse)) return;
  ayde::ElementData eData(baseMemoryAddressOnGPU, threadIndex, 
    elementBlockSize);
  uint64 elementId = eData.GetId();
  ayda::ReducedNumericalModel& model = 
    axis::yabsref<ayda::ReducedNumericalModel>(reducedModelPtr);
  ayde::FiniteElement& e = model.GetElement((size_type)elementId);
  ayde::ElementGeometry& geometry = e.Geometry();
  NLHRFB_GPUFormulation& data = *(NLHRFB_GPUFormulation *)eData.GetFormulationBlock();

  // We will update these guys...
  real *B = data.BMatrix;
  real *nodeCoord = data.NodePosition;

  // Update node coordinate matrix
	for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
	{
		const ayde::Node& node = geometry[nodeIdx];
		nodeCoord[3*nodeIdx + 0] = node.CurrentX();
		nodeCoord[3*nodeIdx + 1] = node.CurrentY();
		nodeCoord[3*nodeIdx + 2] = node.CurrentZ();
	}

  // Determine B-matrix
	for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
	{
		// For the sake of clarity, the expressions below use one-based index so 
		// that it looks more similar to Eq. (79) in Flanagan, Belytschko (1981).
		real B1X = ( YN(2)*( (ZN(6)-ZN(3)) - (ZN(4)-ZN(5)) )  +  YN(3)*(ZN(2)-ZN(4))
			+ YN(4)*( (ZN(3)-ZN(8)) - (ZN(5)-ZN(2)) )  
			+  YN(5)*( (ZN(8)-ZN(6)) - (ZN(2)-ZN(4)) )
			+ YN(6)*(ZN(5)-ZN(2)) + YN(8)*(ZN(4)-ZN(5)) 
			) / (real)12.0;
		real B2X = ( ZN(2)*( (XN(6)-XN(3)) - (XN(4)-XN(5)) )  +  ZN(3)*(XN(2)-XN(4))
			+ ZN(4)*( (XN(3)-XN(8)) - (XN(5)-XN(2)) )  
			+  ZN(5)*( (XN(8)-XN(6)) - (XN(2)-XN(4)) )
			+ ZN(6)*(XN(5)-XN(2)) + ZN(8)*(XN(4)-XN(5)) 
			) / (real)12.0;
		real B3X = ( XN(2)*( (YN(6)-YN(3)) - (YN(4)-YN(5)) )  +  XN(3)*(YN(2)-YN(4))
			+ XN(4)*( (YN(3)-YN(8)) - (YN(5)-YN(2)) )  
			+  XN(5)*( (YN(8)-YN(6)) - (YN(2)-YN(4)) )
			+ XN(6)*(YN(5)-YN(2)) + XN(8)*(YN(4)-YN(5)) 
			) / (real)12.0;
		B[nodeIdx] = B1X; B[8 + nodeIdx] = B2X; B[16 + nodeIdx] = B3X; 
	}

  // Update element volume
	real volume = 0;
	for (int i = 0; i < 8; i++)
	{
		volume += B[i] * nodeCoord[3*i];
	}
	data.Volume = volume;
}


extern void axis::domain::formulations::nlhrfb_commands::RunStrainCommandOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  cudaStream_t stream = (cudaStream_t)streamPtr;
  UpdateNonLinearStrainFBKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

extern 
  void axis::domain::formulations::nlhrfb_commands::RunInternalForceCommandOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  cudaStream_t stream = (cudaStream_t)streamPtr;
  UpdateNonLinearInternalForceFBKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

extern 
  void axis::domain::formulations::nlhrfb_commands::RunUpdateGeometryCommandOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  cudaStream_t stream = (cudaStream_t)streamPtr;
  UpdateNonLinearGeometryFBKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

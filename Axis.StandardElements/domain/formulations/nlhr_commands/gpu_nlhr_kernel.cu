#include "gpu_nlhr_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "../nlhr_gpu_data.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/domain/elements/ElementData.hpp"
#include "yuzu/domain/elements/ElementGeometry.hpp"
#include "yuzu/domain/integration/IntegrationPoint.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ay   = axis::yuzu;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace aydi = axis::yuzu::domain::integration;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

// Derivatives of shape function for one-point quadrature
GPU_ONLY const real dNr[8] = 
{-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125};
GPU_ONLY const real dNs[8] = 
{-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125};
GPU_ONLY const real dNt[8] = 
{-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125};


GPU_ONLY
void CalculateUpdatedJacobianInverse( real *Jinv, real& detJ, 
  const ayde::ElementGeometry& geometry)
{
  // Calculate jacobian matrix. The coefficients below are
  // organized as follows:
  //      [ a  b  c ]
  //  J = [ d  e  f ]
  //      [ g  h  i ]
  real a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0;
  for (int j = 0; j < 8; j++)
  {
    const ayde::Node& node = geometry.GetNode(j);
    a += dNr[j]*node.CurrentX();   b += dNr[j]*node.CurrentY();   
    c += dNr[j]*node.CurrentZ();
    d += dNs[j]*node.CurrentX();   e += dNs[j]*node.CurrentY();   
    f += dNs[j]*node.CurrentZ();
    g += dNt[j]*node.CurrentX();   h += dNt[j]*node.CurrentY();   
    i += dNt[j]*node.CurrentZ();
  }
  detJ = a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;
  Jinv[0] = (e*i - f*h) / detJ;
  Jinv[1] = (c*h - b*i) / detJ;
  Jinv[2] = (b*f - c*e) / detJ;
  Jinv[3] = (f*g - d*i) / detJ;
  Jinv[4] = (a*i - c*g) / detJ;
  Jinv[5] = (c*d - a*f) / detJ;
  Jinv[6] = (d*h - e*g) / detJ;
  Jinv[7] = (b*g - a*h) / detJ;
  Jinv[8] = (a*e - b*d) / detJ;
}


GPU_ONLY
void CalculateDeformationGradient(ayfb::DenseMatrix& deformationGradient,
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
      dui_r += dNr[j] * x;
      dui_s += dNs[j] * x;
      dui_t += dNt[j] * x;
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
  UpdateNonLinearStrainKernel( 
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
  NLHR_GPUFormulation& data = *(NLHR_GPUFormulation *)eData.GetFormulationBlock();
  const real *J0inv = data.InitialJacobianInverse;
  aydi::IntegrationPoint &p = geometry.GetIntegrationPoint(0);
  aydp::InfinitesimalState& state = p.State();

  // Update last deformation gradient
  state.LastDeformationGradient() = state.DeformationGradient();

  // Calculate new deformation gradient
  CalculateDeformationGradient(state.DeformationGradient(), geometry, J0inv);

  // Update element state  
  e.PhysicalState().CopyFrom(state);
}


__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  UpdateNonLinearInternalForceKernel( 
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
  NLHR_GPUFormulation& data = *(NLHR_GPUFormulation *)eData.GetFormulationBlock();
  const real *B = data.BMatrix;
  aydi::IntegrationPoint &p = geometry.GetIntegrationPoint(0);
  aydp::InfinitesimalState& state = p.State();
  const ayfb::ColumnVector &sigma = state.Stress();
  real *internalForce = eData.GetOutputBuffer();

  const real alpha = -8.0 * data.UpdatedJacobianDeterminant;
  // Because of the different layout of our B-matrix, matrix-vector
  // product should be carried this way.
  for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
  {
    const real dNx = B[3*nodeIdx + 0];
    const real dNy = B[3*nodeIdx + 1];
    const real dNz = B[3*nodeIdx + 2];
    const real fx = sigma(0)*dNx + sigma(4)*dNz + sigma(5)*dNy;
    const real fy = sigma(1)*dNy + sigma(3)*dNz + sigma(5)*dNx;
    const real fz = sigma(2)*dNz + sigma(3)*dNy + sigma(4)*dNx;

    internalForce[nodeIdx*3 + 0] = alpha*fx;
    internalForce[nodeIdx*3 + 1] = alpha*fy;
    internalForce[nodeIdx*3 + 2] = alpha*fz;
  }
}


__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  UpdateNonLinearGeometryKernel( 
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
  NLHR_GPUFormulation& data = *(NLHR_GPUFormulation *)eData.GetFormulationBlock();

  // We will update these guys...
  real& detJ = data.UpdatedJacobianDeterminant;
  real *Jinv = data.UpdatedJacobianInverse;
  real *B = data.BMatrix;

  // ...by first calculating the jacobian inverse at the very instant
  CalculateUpdatedJacobianInverse(Jinv, detJ, geometry);

  // ...then, we will use it to calculate the instantaneous B-matrix
  for (int j = 0; j < 8; j++)
  {
    // {dNx} = [J]^(-1)*{dNr}
    real dNjx = Jinv[0]*dNr[j] + Jinv[1]*dNs[j] + Jinv[2]*dNt[j];
    real dNjy = Jinv[3]*dNr[j] + Jinv[4]*dNs[j] + Jinv[5]*dNt[j];
    real dNjz = Jinv[6]*dNr[j] + Jinv[7]*dNs[j] + Jinv[8]*dNt[j];

    // Fill B-matrix for current node
    B[3*j + 0] = dNjx;
    B[3*j + 1] = dNjy;
    B[3*j + 2] = dNjz;
  }
}


extern void axis::domain::formulations::nlhr_commands::RunStrainCommandOnGPU( 
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
  UpdateNonLinearStrainKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

extern 
  void axis::domain::formulations::nlhr_commands::RunInternalForceCommandOnGPU( 
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
  UpdateNonLinearInternalForceKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

extern 
  void axis::domain::formulations::nlhr_commands::RunUpdateGeometryCommandOnGPU( 
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
  UpdateNonLinearGeometryKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

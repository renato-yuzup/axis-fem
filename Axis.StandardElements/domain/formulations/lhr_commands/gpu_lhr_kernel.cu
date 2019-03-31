#include "gpu_lhr_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "../lhr_gpu_data.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/domain/elements/ElementData.hpp"
#include "yuzu/domain/elements/ElementGeometry.hpp"
#include "yuzu/domain/elements/FiniteElement.hpp"
#include "yuzu/domain/integration/IntegrationPoint.hpp"
#include "yuzu/domain/physics/InfinitesimalState.hpp"
#include "yuzu/foundation/blas/vector_operations.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

#define B(x, y)    Bi[24*(x) + (y)]

namespace ay = axis::yuzu;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace aydi = axis::yuzu::domain::integration;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

__device__ void CalculateShapeFunctionDerivatives( real dNr[], real dNs[], real dNt[] )
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

__device__ void CalculateJacobianInverse( real *Jinvi, real& detJ, 
                                         const ayde::ElementGeometry& geometry, const real dNri[], 
                                         const real dNsi[], const real dNti[])
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
    a += dNri[j]*node.X();   b += dNri[j]*node.Y();   c += dNri[j]*node.Z();
    d += dNsi[j]*node.X();   e += dNsi[j]*node.Y();   f += dNsi[j]*node.Z();
    g += dNti[j]*node.X();   h += dNti[j]*node.Y();   i += dNti[j]*node.Z();
  }
  detJ = a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;

  Jinvi[0] = (e*i - f*h) / detJ;
  Jinvi[1] = (c*h - b*i) / detJ;
  Jinvi[2] = (b*f - c*e) / detJ;
  Jinvi[3] = (f*g - d*i) / detJ;
  Jinvi[4] = (a*i - c*g) / detJ;
  Jinvi[5] = (c*d - a*f) / detJ;
  Jinvi[6] = (d*h - e*g) / detJ;
  Jinvi[7] = (b*g - a*h) / detJ;
  Jinvi[8] = (a*e - b*d) / detJ;
}

__device__ void CalculateBMatrix( real *Bi, real *detJ, 
                                 const ayde::ElementGeometry& geometry )
{
  real dNri[8], dNsi[8], dNti[8];
  real Jinvi[9];
  CalculateShapeFunctionDerivatives(dNri, dNsi, dNti);
  CalculateJacobianInverse(Jinvi, *detJ, geometry, dNri, dNsi, dNti);  
  for (int i = 0; i < 144; i++)
  {
    Bi[i] = 0;
  }

  for (int j = 0; j < 8; j++)
  {
    // {dNx} = [J]^(-1)*{dNr}
    real dNjx = Jinvi[0]*dNri[j] + Jinvi[1]*dNsi[j] + Jinvi[2]*dNti[j];
    real dNjy = Jinvi[3]*dNri[j] + Jinvi[4]*dNsi[j] + Jinvi[5]*dNti[j];
    real dNjz = Jinvi[6]*dNri[j] + Jinvi[7]*dNsi[j] + Jinvi[8]*dNti[j];

    // Fill B-matrix for current node
    B(0,3*j) = dNjx;
    B(1,3*j + 1) = dNjy;
    B(2,3*j + 2) = dNjz;
    B(3,3*j) = dNjy;      B(3,3*j + 1) = dNjx;
    B(4,3*j + 1) = dNjz;  B(4,3*j + 2) = dNjy;
    B(5,3*j) = dNjz;                             B(5,3*j + 2) = dNjx;
  }
}

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
UpdateStrainKernel( 
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
  LHR_GPUFormulation& data = *(LHR_GPUFormulation *)eData.GetFormulationBlock();
  real *Bi = data.BMatrix;
  ayfb::ColumnVector& edStrain = e.PhysicalState().LastStrainIncrement();
  ayfb::ColumnVector& eStrain  = e.PhysicalState().Strain();
  aydi::IntegrationPoint &p = geometry.GetIntegrationPoint(0);
  aydp::InfinitesimalState& state = p.State();
  ayfb::ColumnVector& globalDu = model.Kinematics().DisplacementIncrement();

  // calculate strain at the integration point
  edStrain.ClearAll();
  if (!data.InitializedBMatrices)
  {
    CalculateBMatrix(Bi, &data.JacobianDeterminant, geometry);
    data.InitializedBMatrices = true;
  }

  // extract element displacement increment
  real du[24];
  for (int nodeIdx = 0; nodeIdx < 8; nodeIdx++)
  {
    ayde::Node& node = geometry.GetNode(nodeIdx);
    for (int dofIdx = 0; dofIdx < 3; dofIdx++)
    {
      ayde::DoF& dof = node.GetDoF(dofIdx);      
      ayde::Node& parentNode = dof.GetParentNode();
      du[3*nodeIdx + dofIdx] = globalDu(dof.GetId());
    }
  }

  for (int rowIdx = 0; rowIdx < 6; rowIdx++)
  {
    real x = 0;
    for (int i = 0; i < 24; i++)
    {
      x += B(rowIdx, i) * du[i];
    }
    state.LastStrainIncrement()(rowIdx) = x;
  }
  ayfb::VectorSum(state.Strain(), 1.0, state.Strain(), 1.0, 
    state.LastStrainIncrement());

  // update element strain
  edStrain.CopyFrom(state.LastStrainIncrement());
  ayfb::VectorSum(eStrain, 1.0, eStrain, 1.0, edStrain);

}

__global__ void UpdateInternalForceKernel( 
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
  ayde::ElementGeometry& g = e.Geometry();
  LHR_GPUFormulation& data = *(LHR_GPUFormulation *)eData.GetFormulationBlock();
  real *Bi = data.BMatrix;
  if (!data.InitializedBMatrices)
  {
    CalculateBMatrix(Bi, &data.JacobianDeterminant, g);
    data.InitializedBMatrices = true;
  }
  aydi::IntegrationPoint& p = g.GetIntegrationPoint(0);
  aydp::InfinitesimalState& state = p.State();
  ayfb::ColumnVector& stress = state.Stress();
  real *internalForce = eData.GetOutputBuffer();
  for (int i = 0; i < 24; i++)
  {
    real x = 0;
    for (int j = 0; j < 6; j++)
    {
      x += -8.0*data.JacobianDeterminant * B(j,i)*stress(j);
    }
    internalForce[i] = x;
  }
}

extern void axis::domain::formulations::lhr_commands::RunStrainCommandOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr,  uint64 elementBlockSize,
  axis::foundation::memory::RelativePointer& reducedModelPtr, 
  real currentTime, real lastTimeIncrement, real nextTimeIncrement )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  cudaStream_t stream = (cudaStream_t)streamPtr;
  UpdateStrainKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

extern 
  void axis::domain::formulations::lhr_commands::RunInternalForceCommandOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr,  uint64 elementBlockSize,
  axis::foundation::memory::RelativePointer& reducedModelPtr, 
  real currentTime, real lastTimeIncrement, real nextTimeIncrement )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  cudaStream_t stream = (cudaStream_t)streamPtr;
  UpdateInternalForceKernel<<<grid, block, 0, stream>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, elementBlockSize,
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

#include "linear_iso_elastic_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/domain/elements/ElementData.hpp"
#include "yuzu/domain/elements/ElementGeometry.hpp"
#include "yuzu/domain/elements/FiniteElement.hpp"
#include "yuzu/domain/integration/IntegrationPoint.hpp"
#include "yuzu/domain/physics/InfinitesimalState.hpp"
#include "yuzu/domain/physics/UpdatedPhysicalState.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/foundation/blas/vector_operations.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/common/gpu.hpp"

namespace afm = axis::foundation::memory;

namespace ay   = axis::yuzu;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace aydi = axis::yuzu::domain::integration;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;


__device__ void UpdateStress(const real *materialTensor, 
  aydp::UpdatedPhysicalState& state, const aydp::InfinitesimalState& lastState)
{
  for (int i = 0; i < 6; i++)
  {
    real x = 0;
    for (int j = 0; j < 6; j++)
    {
      x += materialTensor[6*i + j] * lastState.LastStrainIncrement()(j);
    }
    state.LastStressIncrement()(i) = x;
    state.Stress()(i) += x;
  }
}

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
RunLinearIsoElasticKernel( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, void * streamPtr, uint64 elementBlockSize, 
  ayfm::RelativePointer reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  using axis::yabsref;
  using axis::yabsptr;
  uint64 baseIndex = 
    ay::GetBaseThreadIndex(gridDim, blockIdx, blockDim, threadIdx);
  if (!ay::IsActiveThread(baseIndex + startIndex, numThreadsToUse)) return;
  ayde::ElementData eData(baseMemoryAddressOnGPU, baseIndex, elementBlockSize); 
  ayda::ReducedNumericalModel& model = 
    yabsref<ayda::ReducedNumericalModel>(reducedModelPtr);
  // get FE geometry data
  uint64 elementId = eData.GetId();
  ayde::FiniteElement &element = model.GetElement((size_type)elementId);  
  ayde::ElementGeometry& g = element.Geometry();
  aydp::InfinitesimalState& elementState = element.PhysicalState();
  const real *materialTensor = eData.MaterialTensor();
  int pointCount = g.GetIntegrationPointCount();
  if (pointCount > 0)
  {
    elementState.Stress().ClearAll();
    elementState.LastStressIncrement().ClearAll();
    for (int pointIdx = 0; pointIdx < pointCount; ++pointIdx)
    {
      aydi::IntegrationPoint& p = g.GetIntegrationPoint(pointIdx);    
      aydp::InfinitesimalState& pointState = p.State();
      // call stress update function
      aydp::UpdatedPhysicalState ups(pointState);
      UpdateStress(materialTensor, ups, pointState);

      // add parcel to element mean stress
      ayfb::VectorSum(elementState.Stress(), 1.0, elementState.Stress(), 1.0, 
        pointState.Stress());
      ayfb::VectorSum(elementState.LastStressIncrement(), 1.0, 
        elementState.LastStressIncrement(), 1.0, 
        pointState.LastStressIncrement());
    }
    // element stress is the average of all parcels
    elementState.Stress().Scale(1.0 / (real)pointCount);
    elementState.LastStressIncrement().Scale(1.0 / (real)pointCount);
  }
  else
  { // formulation does not use integration point;
    // just call stress update function directly on it
    aydp::UpdatedPhysicalState ups(elementState);
    UpdateStress(materialTensor, ups, elementState);
  }
}

extern
void axis::domain::materials::linear_iso_commands::RunLinearIsoElasticOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  RunLinearIsoElasticKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, streamPtr, 
    elementBlockSize, 
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}

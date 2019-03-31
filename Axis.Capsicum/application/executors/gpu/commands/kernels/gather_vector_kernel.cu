#include "gather_vector_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/domain/elements/ElementGeometry.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/common/gpu.hpp"

namespace afm  = axis::foundation::memory;
namespace ay   = axis::yuzu;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
RunGatherVectorKernel(uint64 numThreadsToUse, 
  uint64 startIndex, ayfm::RelativePointer vectorPtr, 
  ayfm::RelativePointer modelPtr)
{
  using axis::yabsref;
  using axis::yabsptr;
  uint64 threadIndex = 
    ay::GetThreadIndex(gridDim, blockIdx, blockDim, threadIdx, startIndex);
  if (!ay::IsActiveThread(threadIndex, numThreadsToUse)) return;
  ayfb::ColumnVector& vector = yabsref<ayfb::ColumnVector>(vectorPtr);
  const ayda::ReducedNumericalModel& model = 
    yabsref<ayda::ReducedNumericalModel>(modelPtr);
  size_type nodeCount = model.GetNodeCount();  
  if (threadIndex >= nodeCount) return;

  const ayde::Node& node = model.GetNode(threadIndex);
  int elementCount = node.GetConnectedElementCount();
  int dofCount = node.GetDofCount();
  // clear entire vector
  for (int dofIdx = 0; dofIdx < dofCount; ++dofIdx)
  {
    const ayde::DoF& dof = node.GetDoF(dofIdx);
    vector(dof.GetId()) = 0;
  }
  for (int eIdx = 0; eIdx < elementCount; ++eIdx)
  {
    ayde::FiniteElement& e = node.GetConnectedElement(eIdx);
    ayde::ElementGeometry& g = e.Geometry();
    const real *elementBucket = model.GetElementOutputBucket(e.GetInternalId());
    int localNodeIdx = g.GetNodeIndex(node);
    for (int dofIdx = 0; dofIdx < dofCount; ++dofIdx)
    {
      const ayde::DoF& dof = node.GetDoF(dofIdx);
      vector(dof.GetId()) += elementBucket[localNodeIdx*dofCount + dofIdx];
    }
  }
}

void axis::application::executors::gpu::commands::kernels::RunGatherVectorOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, 
  afm::RelativePointer& vectorPtr, afm::RelativePointer& modelPtr )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  RunGatherVectorKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreadsToUse, startIndex, 
    reinterpret_cast<ayfm::RelativePointer&>(vectorPtr), 
    reinterpret_cast<ayfm::RelativePointer&>(modelPtr));
}

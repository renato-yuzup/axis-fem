#include "neohookean_kernel.hpp"
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
#include "yuzu/foundation/blas/AutoDenseMatrix.hpp"
#include "yuzu/foundation/blas/AutoSymmetricMatrix.hpp"
#include "yuzu/foundation/blas/linear_algebra.hpp"
#include "yuzu/foundation/blas/matrix_operations.hpp"
#include "yuzu/foundation/blas/vector_operations.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/common/gpu.hpp"
#include "NeoHookeanGPUData.hpp"

namespace afm = axis::foundation::memory;

namespace ay   = axis::yuzu;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace aydi = axis::yuzu::domain::integration;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;
namespace admn = axis::domain::materials::neohookean_commands;

GPU_ONLY void UpdateNeoHookeanStress(aydp::UpdatedPhysicalState& newState,
  const aydp::InfinitesimalState& currentState, real mu, real lambda)
{
  // calculate coefficients
  const ayfb::DenseMatrix& F = currentState.DeformationGradient();  
  const real J = ayfb::Determinant3D(F);
  const real c1 = mu / J;
  const real c2 = (lambda*log(J) - mu) / J;

  // Calculate left Cauchy-Green strain tensor (B -- symmetric)
  ayfb::AutoSymmetricMatrix<3> B;
  ayfb::Product(B, 1.0, F, ayfb::NotTransposed, F, ayfb::Transposed);

  // Calculate Cauchy stress tensor
  ayfb::AutoSymmetricMatrix<3> sigma;
  sigma = B; 
  sigma *= c1;
  sigma(0,0) += c2; sigma(1,1) += c2; sigma(2,2) += c2;

  const real sxx = sigma(0,0);
  const real syy = sigma(1,1);
  const real szz = sigma(2,2);
  const real syz = sigma(1,2);
  const real sxz = sigma(0,2);
  const real sxy = sigma(0,1);

  // Update stress in material point
  ayfb::ColumnVector& updatedStress = newState.Stress();
  updatedStress(0) = sxx;
  updatedStress(1) = syy;
  updatedStress(2) = szz;
  updatedStress(3) = syz;
  updatedStress(4) = sxz;
  updatedStress(5) = sxy;
}

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  RunNeoHookeanKernel( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, void * streamPtr, uint64 elementBlockSize, 
  ayfm::RelativePointer reducedModelPtr )
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
  admn::NeoHookeanGPUData* matData = 
    (admn::NeoHookeanGPUData *)eData.GetMaterialBlock();
  real lambda = matData->LambdaCoefficient;
  real mu = matData->MuCoefficient;
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
      UpdateNeoHookeanStress(ups, pointState, mu, lambda);

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
    UpdateNeoHookeanStress(ups, elementState, mu, lambda);
  }
}

extern void axis::domain::materials::neohookean_commands::RunNeoHookeanOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  RunNeoHookeanKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, streamPtr, 
    elementBlockSize, 
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr));
}


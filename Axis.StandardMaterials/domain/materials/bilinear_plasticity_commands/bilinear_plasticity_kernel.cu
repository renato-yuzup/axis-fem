#include "bilinear_plasticity_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/domain/elements/ElementData.hpp"
#include "yuzu/domain/elements/ElementGeometry.hpp"
#include "yuzu/domain/elements/FiniteElement.hpp"
#include "yuzu/domain/integration/IntegrationPoint.hpp"
#include "yuzu/domain/physics/InfinitesimalState.hpp"
#include "yuzu/domain/physics/UpdatedPhysicalState.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/foundation/blas/AutoColumnVector.hpp"
#include "yuzu/foundation/blas/matrix_operations.hpp"
#include "yuzu/foundation/blas/vector_operations.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/common/gpu.hpp"
#include "yuzu/mechanics/continuum.hpp"
#include "BiLinearPlasticityGPUData.hpp"

namespace admb = axis::domain::materials::bilinear_plasticity_commands;
namespace afm  = axis::foundation::memory;

namespace ay   = axis::yuzu;
namespace aym  = axis::yuzu::mechanics;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace aydi = axis::yuzu::domain::integration;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY void CalculateStressRate( ayfb::AutoSymmetricMatrix<3>& stressRate, 
  const real *Ce,
  const ayfb::AutoSymmetricMatrix<3>& stressTensor,
  const ayfb::AutoSymmetricMatrix<3>& rateOfDeformationTensor, 
  const ayfb::AutoDenseMatrix<3,3>& spinTensor )
{
  const ayfb::AutoSymmetricMatrix<3>& D = rateOfDeformationTensor;
  const ayfb::AutoDenseMatrix<3,3>& W = spinTensor;
  ayfb::AutoColumnVector<6> Dvec, jaumannVec;
  ayfb::TransformSecondTensorToVoigt(Dvec, D);
  Dvec(3) *= 2; Dvec(4) *= 2; Dvec(5) *= 2; 

  // Calculate Jaumann objective stress rate as a function of the rate of 
  // deformation tensor
  for (int rowIdx = 0; rowIdx < 6; rowIdx++)
  {
    real x = 0;
    for (int colIdx = 0; colIdx < 6; colIdx++)
    {
      real cij = Ce[rowIdx*6 + colIdx];
      x += cij * Dvec(colIdx);
    }
    jaumannVec(rowIdx) = x;
  }
  ayfb::TransformVoigtToSecondTensor(stressRate, jaumannVec);

  // Add to stress rate contribution regarding rigid body rotations
  ayfb::AccumulateProduct(stressRate, +1.0, W, stressTensor);
  ayfb::AccumulateProduct(stressRate, -1.0, stressTensor, W);
}

GPU_ONLY void UpdateElasticStress( aydp::UpdatedPhysicalState& newState, 
  const aydp::InfinitesimalState& currentState, 
  const real *materialTensor,
  const ayfb::AutoSymmetricMatrix<3>& rateOfDeformationTensor,
  const ayfb::AutoDenseMatrix<3,3>& spinTensor, real dt )
{
  const ayfb::AutoSymmetricMatrix<3>& D = rateOfDeformationTensor;
  const ayfb::AutoDenseMatrix<3,3>&   W = spinTensor; 
  ayfb::AutoSymmetricMatrix<3> stressRate;    // stress rate (wrt time)
  ayfb::AutoSymmetricMatrix<3> curSigma;      // total stress at current step

  // Write current stress in matrix form
  ayfb::TransformVoigtToSecondTensor(curSigma, currentState.Stress());

  // Calculate stress rate using objective rates
  CalculateStressRate(stressRate, materialTensor, curSigma, 
    rateOfDeformationTensor, spinTensor);

  // Calculate stress increment, but do not increment yet, as the increment
  // parcel will be corrected by the plasticity step
  ayfb::ColumnVector& dSigma = newState.LastStressIncrement();
  ayfb::TransformSecondTensorToVoigt(dSigma, stressRate);
  dSigma *= dt; // dSigma = stressRate * dt
}

GPU_ONLY void ExecutePlasticCorrection( aydp::UpdatedPhysicalState& newState, 
  const aydp::InfinitesimalState& currentState, real H, real yieldStress,
  const real *Ce, ayfb::AutoSymmetricMatrix<3>& D, const ayfb::AutoDenseMatrix<3,3>& W,
  real dt )
{
  ayfb::AutoColumnVector<6> trialStressVec;
  const ayfb::ColumnVector& sigma0 = newState.Stress();       
  ayfb::ColumnVector& dSigma = newState.LastStressIncrement();
  ayfb::AutoSymmetricMatrix<3> trialStress;  
  ayfb::AutoSymmetricMatrix<3> flowDirection;
  ayfb::AutoColumnVector<6> flowDirectionVec;
  real dLambda = 0;                     

  // Calculate total trial deviatoric stress
  ayfb::VectorSum(trialStressVec, 1.0, sigma0, 1.0, dSigma);
  ayfb::TransformVoigtToSecondTensor(trialStress, trialStressVec);
  real pressure = trialStress.Trace() / 3.0;
  trialStress(0,0) -= pressure; 
  trialStress(1,1) -= pressure; 
  trialStress(2,2) -= pressure;

  // Determine yield function
  real equivStress = ayfb::DoubleContraction(1.0, trialStress, 1.0, trialStress);
  equivStress = sqrt(1.5 * equivStress);
  real equivPlasticStrain = currentState.EffectivePlasticStrain();
  real yieldFun = equivStress - yieldStress - H*equivPlasticStrain;

  // Check consistency condition
  if (yieldFun > 0)   // plastic step occurring, correct stress 
  {                   // and update plastic strain state
    // Determine flow direction
    flowDirection = trialStress;
    flowDirection *= (3.0 / (2.0*equivStress));
    ayfb::TransformSecondTensorToVoigt(flowDirectionVec, flowDirection);
    flowDirectionVec(3) *= 2.0; 
    flowDirectionVec(4) *= 2.0; 
    flowDirectionVec(5) *= 2.0; 

    // Calculate plastic multiplier
    ayfb::AutoColumnVector<6> aux, aux2;
    ayfb::TransformSecondTensorToVoigt(aux, D);
    aux(3) *= 2.0; aux(4) *= 2.0; aux(5) *= 2.0; 
    for (int rowIdx = 0; rowIdx < 6; rowIdx++)
    {
      real x = 0;
      for (int colIdx = 0; colIdx < 6; colIdx++)
      {
        x += Ce[rowIdx*6 + colIdx] * aux(colIdx);
      }
      aux2(rowIdx) = x;
    }
    real numerator = ayfb::VectorScalarProduct(flowDirectionVec, aux2);

    for (int rowIdx = 0; rowIdx < 6; rowIdx++)
    {
      real x = 0;
      for (int colIdx = 0; colIdx < 6; colIdx++)
      {
        x += Ce[rowIdx*6 + colIdx] * flowDirectionVec(colIdx);
      }
      aux(rowIdx) = x;
    }
    real denominator = ayfb::VectorScalarProduct(flowDirectionVec, aux);
    denominator += H;
    dLambda = abs(numerator / denominator);

    // Determine plastic strain increment (explicit integration)
    ayfb::AutoColumnVector<6> dPlasticStrain;
    dPlasticStrain = flowDirectionVec;
    dPlasticStrain *= dLambda*dt;
    ayfb::VectorSum(newState.PlasticStrain(), 1.0, newState.PlasticStrain(), 1.0, dPlasticStrain);

    // Update effective plastic strain (explicit integration)
    newState.EffectivePlasticStrain() += dLambda*dt;

    // Calculate rate of plastic deformation
    ayfb::AutoSymmetricMatrix<3> Dp;
    Dp = flowDirection; Dp *= dLambda;

    // Recalculate stress rate    
    ayfb::AutoSymmetricMatrix<3>& De = D; // just to keep notation
    ayfb::Sum(De, 1.0, De, -1.0, Dp);
    ayfb::AutoSymmetricMatrix<3> sigmaTensor;
    ayfb::AutoSymmetricMatrix<3> stressRate;
    ayfb::TransformVoigtToSecondTensor(sigmaTensor, sigma0);
    CalculateStressRate(stressRate, Ce, sigmaTensor, De, W);

    // Recalculate stress increment
    ayfb::TransformSecondTensorToVoigt(dSigma, stressRate);
    dSigma *= dt;
  }
}

GPU_ONLY void UpdateBiLinearPlasticityStress(const real *Ce, 
  aydp::UpdatedPhysicalState& newState, 
  const aydp::InfinitesimalState& currentState, real yieldStress, 
  real hardeningCoefficient, real dt)
{
  ayfb::AutoDenseMatrix<3,3> L;         // velocity gradient
  ayfb::AutoSymmetricMatrix<3> D;       // deformation rate tensor
  ayfb::AutoDenseMatrix<3,3> W;         // spin tensor
  const ayfb::DenseMatrix& Fnext = currentState.DeformationGradient();
  const ayfb::DenseMatrix& Fcur = currentState.LastDeformationGradient();

  // Calculate rate of deformation and spin tensors
  aym::CalculateVelocityGradient(L, Fnext, Fcur, dt);
  ayfb::DecomposeSymmetric(D, L);
  ayfb::DecomposeSkew(W, L);

  // Do trial elastic step
  UpdateElasticStress(newState, currentState, Ce, D, W, dt);

  // Execute plastic correction, if needed
  ExecutePlasticCorrection(newState, currentState, hardeningCoefficient, 
    yieldStress, Ce, D, W, dt);

  // Increment stress
  ayfb::ColumnVector& sigma = newState.Stress();
  ayfb::ColumnVector& dSigma = newState.LastStressIncrement();
  sigma += dSigma;
}

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  RunBiLinearPlasticityKernel( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, void * streamPtr, uint64 elementBlockSize, 
  ayfm::RelativePointer reducedModelPtr, real dt )
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
  admb::BiLinearPlasticityGPUData& matData = 
    *(admb::BiLinearPlasticityGPUData *)eData.GetMaterialBlock();
  real yieldStress = matData.YieldStress;
  real H = matData.HardeningCoefficient;
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
      UpdateBiLinearPlasticityStress(materialTensor, ups, pointState, 
        yieldStress, H, dt);

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
    UpdateBiLinearPlasticityStress(materialTensor, ups, elementState, 
      yieldStress, H, dt);
  }
}


extern void axis::domain::materials::bilinear_plasticity_commands::
  RunBiLinearPlasticityOnGPU( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize, 
  axis::foundation::memory::RelativePointer& reducedModelPtr, 
  real nextTimeIncrement )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  RunBiLinearPlasticityKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, streamPtr, 
    elementBlockSize, 
    reinterpret_cast<ayfm::RelativePointer&>(reducedModelPtr), 
    nextTimeIncrement);
}

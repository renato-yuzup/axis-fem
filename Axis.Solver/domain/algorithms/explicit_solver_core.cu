#include "explicit_solver_core.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "Dimension3D.hpp"
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/blas/SubColumnVector.hpp"
#include "foundation/memory/pointer.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/common/gpu.hpp"
#include "dof_status.hpp"

namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;
namespace afm = axis::foundation::memory;

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
ExplicitSolverBeforeKernel(uint64 numThreads, uint64 startIndex, 
  ayfm::RelativePointer reducedModelPtr, ayfm::RelativePointer gpuMaskPtr, 
  real t, real dt, long iterationIndex)
{
	/**************************************************************************************
	/*    CENTRAL DIFFERENCE METHOD ALGORITHM MAIN LOOP (BEFORE ELEMENT UPDATE)
	**************************************************************************************/
	// initialize work vectors
	ayda::ReducedNumericalModel& model = 
    axis::yabsref<ayda::ReducedNumericalModel>(reducedModelPtr);
  ayda::ModelDynamics& dynamics     = model.Dynamics();
  ayda::ModelKinematics& kinematics = model.Kinematics();
  ayfb::ColumnVector& u     = kinematics.Displacement();
  ayfb::ColumnVector& v     = kinematics.Velocity();
  ayfb::ColumnVector& a     = kinematics.Acceleration();
  ayfb::ColumnVector& du    = kinematics.DisplacementIncrement();
  ayfb::ColumnVector& fR    = dynamics.ReactionForce();
  ayfb::ColumnVector& fInt  = dynamics.InternalForces();
  ayfb::ColumnVector& fExt  = dynamics.ExternalLoads();
  const char *dofMask       = axis::yabsptr<char>(gpuMaskPtr);

  uint64 index = axis::yuzu::GetThreadIndex(gridDim, blockIdx, blockDim, 
    threadIdx, startIndex);
  if (!axis::yuzu::IsActiveThread(index, numThreads)) return;

	// 1) Calculate next displacement increment and update total displacement
  if (dofMask[index] == DOF_STATUS_PRESCRIBED_DISPLACEMENT)
  { // If it is a prescribed displacement, the displacement has already been
    // updated by the boundary condition; simply calculate the given increment
    // for coherence.
    du(index) = u(index) - du(index);
  }
  else
  {
    if (dofMask[index] == DOF_STATUS_FREE || dofMask[index] == DOF_STATUS_PRESCRIBED_VELOCITY)
    {
      du(index) = v(index) * dt;
    }
    u(index) += du(index);
  }
	
  // 2) Update node coordinates
  uint64 nodeCount = model.GetNodeCount();
  if (index < nodeCount)
  {
    ayde::Node& node = model.GetNode(index);
    uint64 dofXId = node.GetDoF(0).GetId();
    uint64 dofYId = node.GetDoF(1).GetId();
    uint64 dofZId = node.GetDoF(2).GetId();
    node.CurrentX() += du(dofXId);
    node.CurrentY() += du(dofYId);
    node.CurrentZ() += du(dofZId);
  }
}

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
ExplicitSolverAfterKernel(uint64 numThreads, uint64 startIndex, 
  ayfm::RelativePointer reducedModelPtr, ayfm::RelativePointer massMatrix,
  ayfm::RelativePointer gpuMaskPtr, real t, real dt, long iterationIndex)
{
	/**************************************************************************************
	/*    CENTRAL DIFFERENCE METHOD ALGORITHM MAIN LOOP (AFTER ELEMENT UPDATE)
	**************************************************************************************/
	// initialize work vectors
	ayda::ReducedNumericalModel& model = 
    axis::yabsref<ayda::ReducedNumericalModel>(reducedModelPtr);
  ayda::ModelDynamics& dynamics     = model.Dynamics();
  ayda::ModelKinematics& kinematics = model.Kinematics();
  ayfb::ColumnVector& u       = kinematics.Displacement();
  ayfb::ColumnVector& v       = kinematics.Velocity();
  ayfb::ColumnVector& a       = kinematics.Acceleration();
  ayfb::ColumnVector& du = kinematics.DisplacementIncrement();
  ayfb::ColumnVector& fR      = dynamics.ReactionForce();
  ayfb::ColumnVector& fInt    = dynamics.InternalForces();
  ayfb::ColumnVector& fExt    = dynamics.ExternalLoads();
  ayfb::ColumnVector& m       = axis::yabsref<ayfb::ColumnVector>(massMatrix);
  const char *dofMask = axis::yabsptr<char>(gpuMaskPtr);

  uint64 index = axis::yuzu::GetThreadIndex(gridDim, blockIdx, blockDim, 
    threadIdx, startIndex);
  if (!axis::yuzu::IsActiveThread(index, numThreads)) return;
  const char dofVal = dofMask[index];

  // 1) Calculate effective nodal force
  real fRx = fExt(index) + fInt(index);
  fR(index) = fRx;

  // 2) Calculate accelerations
  if (dofVal != DOF_STATUS_LOCKED && dofVal != DOF_STATUS_PRESCRIBED_VELOCITY)
  {
    a(index) = fRx / m(index);
  }

  // 3) Update velocities
  if (dofVal == DOF_STATUS_FREE || dofVal == DOF_STATUS_PRESCRIBED_DISPLACEMENT)
  {
    v(index) += a(index) * dt;
  }

  // 4) If this is a prescribed displacement, we will use the corresponding
  // displacement increment index to store current displacement, so that in
  // the next step, we can calculate the increment.
  if (dofMask[index] == DOF_STATUS_PRESCRIBED_DISPLACEMENT)
  {
    du(index) = u(index);
  }
}

extern void axis::domain::algorithms::RunExplicitBeforeStepOnGPU(
  uint64 numThreads, uint64 startIndex, afm::RelativePointer modelPtr,
  afm::RelativePointer gpuMaskPtr, real t, real dt, long iterationIndex, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr)
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  ExplicitSolverBeforeKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreads, startIndex, 
    reinterpret_cast<ayfm::RelativePointer&>(modelPtr), 
    reinterpret_cast<ayfm::RelativePointer&>(gpuMaskPtr), 
    t, dt, iterationIndex);
}

extern void axis::domain::algorithms::RunExplicitAfterStepOnGPU(
  uint64 numThreads, uint64 startIndex, afm::RelativePointer modelPtr,
  afm::RelativePointer lumpedMassPtr, afm::RelativePointer gpuMaskPtr, 
  real t, real dt, long iterationIndex, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr)
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  ExplicitSolverAfterKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreads, startIndex, 
    reinterpret_cast<ayfm::RelativePointer&>(modelPtr), 
    reinterpret_cast<ayfm::RelativePointer&>(lumpedMassPtr), 
    reinterpret_cast<ayfm::RelativePointer&>(gpuMaskPtr), 
    t, dt, iterationIndex);
}

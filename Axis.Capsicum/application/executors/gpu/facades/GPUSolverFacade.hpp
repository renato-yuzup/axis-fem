#pragma once
#include "domain/algorithms/ExternalSolverFacade.hpp"
#include "services/scheduling/gpu/GPUTask.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"
#include "domain/boundary_conditions/BoundaryConditionUpdateCommand.hpp"
#include "domain/curves/CurveUpdateCommand.hpp"
#include <list>

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace facades {

/**
 * Implements a solver auxiliary processing object that executes on GPU.
 */
class GPUSolverFacade : public axis::domain::algorithms::ExternalSolverFacade
{
public:
  GPUSolverFacade(void);
  ~GPUSolverFacade(void);

  /**
   * Sets task to be used when dispatching solver-specific operations to GPU.
   *
   * @param [in,out] task The task.
   */
  void SetSolverTask(axis::services::scheduling::gpu::GPUTask& task);

  /**
   * Adds a boundary condition task which defines a set of boundary conditions
   * (necessarily of the same class type) declared in the numerical model. 
   * These tasks will be used when referring to boundary conditions value 
   * updates.
   *
   * @param type             The boundary condition type.
   * @param [in,out] task    The task representing the set of boundary 
   *                         conditions.
   * @param [in,out] command The command to be issued when updating
   *                 boundary condition values.
   * @param blockSize        Block length of boundary condition data (only
   *                         custom data applies).
   */
  void AddBoundaryConditionTask(
    axis::domain::boundary_conditions::BoundaryCondition::ConstraintType type,
    axis::services::scheduling::gpu::GPUTask& task,
    axis::domain::boundary_conditions::BoundaryConditionUpdateCommand& command,
    size_type blockSize);

  /**
   * Adds a curve task which defines a set of curves (necessarily of the same 
   * class type) declared in the numerical model. These tasks will be used when
   * referring to curve value updates.
   *
   * @param [in,out] task    The task representing the set of curves.
   * @param [in,out] command The command to be issued when updating curve 
   *                 values.
   * @param blockSize        Block length of curve data (only custom data 
   *                         applies).
   */
  void AddCurveTask(axis::services::scheduling::gpu::GPUTask& task,
    axis::domain::curves::CurveUpdateCommand& command, size_type blockSize);

  /**
   * Allocates memory, in the host and external processing device, for model
   * objects and any supplementary information needed.
   */
  void AllocateMemory(void);

  /**
   * Deallocates memory claimed by this facade.
   */
  void DeallocateMemory(void);

  /**
   * Initializes model memory and any metadata required to correctly run 
   * operations in this facade.
   */
  void InitMemory(void);

  /**
   * Mirrors model memory to external processing device.
   */
  void Mirror(void);

  /**
   * Copies model memory in the external processing device back to the host
   * memory, so that modifications made in the external device are reflected
   * in the host.
   */
  void Restore(void);

  /**
   * Marks a host memory block for use by the external processing device. In
   * other words, it marks the block as non-pageable.
   *
   * @param [in,out] baseAddress Base address of the memory block.
   * @param blockSize            Size of the block.
   */
  void ClaimMemory(void *baseAddress, uint64 blockSize);

  /**
   * Disables a host memory block for use by the external processing device. In
   * other words, it marks the block back as pageable.
   *
   * @param [in,out] baseAddress Base address of the memory block.
   */
  void ReturnMemory(void *baseAddress);

  /**
   * Returns the base address of the model memory in GPU.
   *
   * @return null if it fails, else the GPU model arena address.
   */
  void *GetGPUModelArenaAddress(void) const;

  virtual void UpdateCurves(real time);
  virtual void UpdateAccelerations(
    axis::foundation::memory::RelativePointer& globalAccelerationVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask);
  virtual void UpdateVelocities(
    axis::foundation::memory::RelativePointer& globalVelocityVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask);
  virtual void UpdateDisplacements(
    axis::foundation::memory::RelativePointer& globalDisplacementVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask);
  virtual void UpdateExternalLoads(
    axis::foundation::memory::RelativePointer& externalLoadVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask);
  virtual void UpdateLocks(
    axis::foundation::memory::RelativePointer& globalDisplacementVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask);
  virtual void RunKernel( axis::foundation::computing::KernelCommand& kernel );
  virtual void GatherVector(axis::foundation::memory::RelativePointer& vectorPtr,
    axis::foundation::memory::RelativePointer& modelPtr);
  virtual void Synchronize(void);
private:
  template <class T>
  class Tuple
  {
  public:
    Tuple(axis::services::scheduling::gpu::GPUTask& bcTask,
      T& command,
      size_type blockSize);

    axis::services::scheduling::gpu::GPUTask *Task;
    T *GPUCommand;
    size_type BlockSize;
  };

  typedef Tuple<axis::domain::boundary_conditions::BoundaryConditionUpdateCommand> bc_tuple;
  typedef Tuple<axis::domain::curves::CurveUpdateCommand> curve_tuple;
  typedef std::list<bc_tuple> bc_task_list;
  typedef std::list<curve_tuple> curve_task_list;

  void UpdateBoundaryCondition(bc_task_list& bcTasks, 
    axis::foundation::memory::RelativePointer& targetVector,
    real time, axis::foundation::memory::RelativePointer& vectorMask,
    bool ignoreMask);
  template <class T> void AllocateTaskListMemory(std::list<Tuple<T>>& tasks);
  template <class T> void DeallocateTaskListMemory(std::list<Tuple<T>>& tasks);
  template <class T> void InitTaskListMemory(std::list<Tuple<T>>& tasks);
  template <class T> void MirrorTaskListMemory(std::list<Tuple<T>>& tasks);
  template <class T> void RestoreTaskListMemory(std::list<Tuple<T>>& tasks);
  template <class T> void DeleteTaskList(std::list<Tuple<T>>& tasks);
  virtual void PrepareForCollectionRound( axis::domain::analyses::ReducedNumericalModel& model );

  axis::services::scheduling::gpu::GPUTask *solverTask_;
  curve_task_list curves_;
  bc_task_list accelerationBcs_;
  bc_task_list velocityBcs_;
  bc_task_list displacementBcs_;
  bc_task_list loadsBcs_;
  bc_task_list locksBcs_;
};

} } } } } // namespace axis::application::executors::gpu::facades

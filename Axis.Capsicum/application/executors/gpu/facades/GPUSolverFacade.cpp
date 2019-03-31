#include "GPUSolverFacade.hpp"
#include "application/executors/gpu/commands/GatherVectorCommand.hpp"
#include "application/executors/gpu/commands/PushBcToVectorCommand.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/analyses/ReducedNumericalModel.hpp"
#include "domain/analyses/ModelOperatorFacade.hpp"
#include "foundation/memory/pointer.hpp"

namespace aaegc = axis::application::executors::gpu::commands;
namespace aaegf = axis::application::executors::gpu::facades;
namespace ada   = axis::domain::analyses;
namespace adbc  = axis::domain::boundary_conditions;
namespace adc   = axis::domain::collections;
namespace adcu  = axis::domain::curves;
namespace ade   = axis::domain::elements;
namespace assg  = axis::services::scheduling::gpu;
namespace afc   = axis::foundation::computing;
namespace afm   = axis::foundation::memory;

aaegf::GPUSolverFacade::GPUSolverFacade(void)
{
  solverTask_ = nullptr;
}

aaegf::GPUSolverFacade::~GPUSolverFacade(void)
{
  delete solverTask_;
  DeleteTaskList(curves_);
  DeleteTaskList(accelerationBcs_);
  DeleteTaskList(velocityBcs_);
  DeleteTaskList(displacementBcs_);
  DeleteTaskList(loadsBcs_);
  DeleteTaskList(locksBcs_);
}

void aaegf::GPUSolverFacade::SetSolverTask( assg::GPUTask& task )
{
  if (solverTask_ != nullptr && solverTask_ != &task)
  {
    delete solverTask_;
  }
  solverTask_ = &task;
}

void aaegf::GPUSolverFacade::AddBoundaryConditionTask( 
  adbc::BoundaryCondition::ConstraintType type, assg::GPUTask& task,
  adbc::BoundaryConditionUpdateCommand& command, size_type blockSize)
{
  bc_tuple tuple(task, command, blockSize);
  switch (type)
  {
  case adbc::BoundaryCondition::PrescribedDisplacement:
    displacementBcs_.push_back(tuple);
    break;
  case adbc::BoundaryCondition::PrescribedVelocity:
    velocityBcs_.push_back(tuple);
    break;
  case adbc::BoundaryCondition::NodalLoad:
    loadsBcs_.push_back(tuple);
    break;
  case adbc::BoundaryCondition::Lock:
    locksBcs_.push_back(tuple);
    break;
  default:
    assert(!"Unknown boundary condition type!");
    break;
  }
}

void aaegf::GPUSolverFacade::AddCurveTask( assg::GPUTask& task,
  adcu::CurveUpdateCommand& command, size_type blockSize)
{
  curves_.push_back(curve_tuple(task, command, blockSize));
}

void aaegf::GPUSolverFacade::AllocateMemory( void )
{
  solverTask_->AllocateMemory();
  AllocateTaskListMemory(curves_);
  AllocateTaskListMemory(accelerationBcs_);
  AllocateTaskListMemory(velocityBcs_);
  AllocateTaskListMemory(displacementBcs_);
  AllocateTaskListMemory(loadsBcs_);
  AllocateTaskListMemory(locksBcs_);
}

void aaegf::GPUSolverFacade::DeallocateMemory( void )
{
  solverTask_->DeallocateMemory();
  DeallocateTaskListMemory(curves_);
  DeallocateTaskListMemory(accelerationBcs_);
  DeallocateTaskListMemory(velocityBcs_);
  DeallocateTaskListMemory(displacementBcs_);
  DeallocateTaskListMemory(loadsBcs_);
  DeallocateTaskListMemory(locksBcs_);
}

void aaegf::GPUSolverFacade::InitMemory( void )
{
  solverTask_->InitMemory();
  InitTaskListMemory(curves_);
  InitTaskListMemory(accelerationBcs_);
  InitTaskListMemory(velocityBcs_);
  InitTaskListMemory(displacementBcs_);
  InitTaskListMemory(loadsBcs_);
  InitTaskListMemory(locksBcs_);
}

void aaegf::GPUSolverFacade::Mirror( void )
{
  solverTask_->Mirror();
  MirrorTaskListMemory(curves_);
  MirrorTaskListMemory(accelerationBcs_);
  MirrorTaskListMemory(velocityBcs_);
  MirrorTaskListMemory(displacementBcs_);
  MirrorTaskListMemory(loadsBcs_);
  MirrorTaskListMemory(locksBcs_);
}

void aaegf::GPUSolverFacade::Restore( void )
{
  solverTask_->Restore();
  RestoreTaskListMemory(curves_);
  RestoreTaskListMemory(accelerationBcs_);
  RestoreTaskListMemory(velocityBcs_);
  RestoreTaskListMemory(displacementBcs_);
  RestoreTaskListMemory(loadsBcs_);
  RestoreTaskListMemory(locksBcs_);
}

void aaegf::GPUSolverFacade::ClaimMemory( void *baseAddress, uint64 blockSize )
{
  solverTask_->ClaimMemory(baseAddress, blockSize);
}

void aaegf::GPUSolverFacade::ReturnMemory( void *baseAddress )
{
  solverTask_->ReturnMemory(baseAddress);
}

void * aaegf::GPUSolverFacade::GetGPUModelArenaAddress( void ) const
{
  return solverTask_->GetDeviceMemoryAddress(0);
}

void aaegf::GPUSolverFacade::UpdateCurves( real time )
{
  curve_task_list::iterator end = curves_.end();
  for (curve_task_list::iterator it = curves_.begin(); it != end; ++it)
  { // run asynchronously...
    adcu::CurveUpdateCommand& curveCmd = *it->GPUCommand;
    assg::GPUTask& curveTask = *it->Task;
    curveCmd.SetTime(time);
    curveTask.RunCommand(curveCmd);
  }

  for (curve_task_list::iterator it = curves_.begin(); it != end; ++it)
  { // ...and then synchronize
    assg::GPUTask& curveTask = *it->Task;
    curveTask.Synchronize();
  }
}

void aaegf::GPUSolverFacade::UpdateAccelerations(
  afm::RelativePointer& globalAccelerationVector, real time,
  afm::RelativePointer& vectorMask)
{
  UpdateBoundaryCondition(accelerationBcs_, globalAccelerationVector, time, 
    vectorMask, false);
}

void aaegf::GPUSolverFacade::UpdateVelocities(
  afm::RelativePointer& globalVelocityVector, real time,
  afm::RelativePointer& vectorMask)
{
  UpdateBoundaryCondition(velocityBcs_, globalVelocityVector, time, 
    vectorMask, false);
}

void aaegf::GPUSolverFacade::UpdateDisplacements(
  afm::RelativePointer& globalDisplacementVector, real time,
  afm::RelativePointer& vectorMask)
{
  UpdateBoundaryCondition(displacementBcs_, globalDisplacementVector,
    time, vectorMask, false);
}

void aaegf::GPUSolverFacade::UpdateExternalLoads(
  afm::RelativePointer& externalLoadVector, real time,
  axis::foundation::memory::RelativePointer& vectorMask)
{
  UpdateBoundaryCondition(loadsBcs_, externalLoadVector, time, 
    vectorMask, true);
}

void aaegf::GPUSolverFacade::UpdateLocks(
  afm::RelativePointer& globalDisplacementVector, real time,
  afm::RelativePointer& vectorMask)
{
  UpdateBoundaryCondition(locksBcs_, globalDisplacementVector, time, 
    vectorMask, false);
}

void aaegf::GPUSolverFacade::RunKernel( afc::KernelCommand& kernel )
{
  solverTask_->RunCommand(kernel);
}

void aaegf::GPUSolverFacade::GatherVector( afm::RelativePointer& vectorPtr, 
                                          afm::RelativePointer& modelPtr )
{
  aaegc::GatherVectorCommand gatherCmd(modelPtr, vectorPtr);
  solverTask_->RunCommand(gatherCmd);
}

void aaegf::GPUSolverFacade::Synchronize( void )
{
  solverTask_->Synchronize();
}

void aaegf::GPUSolverFacade::UpdateBoundaryCondition( bc_task_list& bcTasks,
  afm::RelativePointer& targetVector, real time, 
  afm::RelativePointer& vectorMask, bool ignoreMask)
{
  bc_task_list::iterator end = bcTasks.end();
  for (bc_task_list::iterator it = bcTasks.begin(); it != end; ++it)
  { // run asynchronously...
    adbc::BoundaryConditionUpdateCommand& bcCmd = *it->GPUCommand;
    bcCmd.Configure(time, vectorMask);
    assg::GPUTask& bcTask = *it->Task;
    bcTask.RunCommand(bcCmd);
  }

  for (bc_task_list::iterator it = bcTasks.begin(); it != end; ++it)
  { // ...and then synchronize
    assg::GPUTask& bcTask = *it->Task;
    bcTask.Synchronize();
  }

  for (bc_task_list::iterator it = bcTasks.begin(); it != end; ++it)
  { // gather results into the global vector
    assg::GPUTask& bcTask = *it->Task;
    aaegc::PushBcToVectorCommand gatherCmd(targetVector, vectorMask, ignoreMask, it->BlockSize);
    bcTask.RunCommand(gatherCmd);
  }

  for (bc_task_list::iterator it = bcTasks.begin(); it != end; ++it)
  { // ...and then synchronize again
    assg::GPUTask& bcTask = *it->Task;
    bcTask.Synchronize();
  }
}

template <class T>
void aaegf::GPUSolverFacade::AllocateTaskListMemory(std::list<Tuple<T>>& tasks)
{
  typedef std::list<Tuple<T>> task_list;
  task_list::iterator end = tasks.end();
  for (task_list::iterator it = tasks.begin(); it != end; ++it)
  {
    (*it).Task->AllocateMemory();
  }
}

template <class T>
void aaegf::GPUSolverFacade::DeallocateTaskListMemory(std::list<Tuple<T>>& tasks)
{
  typedef std::list<Tuple<T>> task_list;
  task_list::iterator end = tasks.end();
  for (task_list::iterator it = tasks.begin(); it != end; ++it)
  {
    (*it).Task->DeallocateMemory();
  }
}

template <class T>
void aaegf::GPUSolverFacade::InitTaskListMemory(std::list<Tuple<T>>& tasks)
{
  typedef std::list<Tuple<T>> task_list;
  task_list::iterator end = tasks.end();
  for (task_list::iterator it = tasks.begin(); it != end; ++it)
  {
    (*it).Task->InitMemory();
  }
}

template <class T>
void aaegf::GPUSolverFacade::MirrorTaskListMemory(std::list<Tuple<T>>& tasks)
{
  typedef std::list<Tuple<T>> task_list;
  task_list::iterator end = tasks.end();
  for (task_list::iterator it = tasks.begin(); it != end; ++it)
  {
    (*it).Task->Mirror();
  }
}

template <class T>
void aaegf::GPUSolverFacade::RestoreTaskListMemory(std::list<Tuple<T>>& tasks)
{
  typedef std::list<Tuple<T>> task_list;
  task_list::iterator end = tasks.end();
  for (task_list::iterator it = tasks.begin(); it != end; ++it)
  {
    (*it).Task->Restore();
  }
}

template <class T>
void aaegf::GPUSolverFacade::DeleteTaskList( std::list<Tuple<T>>& tasks )
{
  typedef std::list<Tuple<T>> task_list;
  task_list::iterator end = tasks.end();
  for (task_list::iterator it = tasks.begin(); it != end; ++it)
  {
   delete (*it).Task;
  }
}

void aaegf::GPUSolverFacade::PrepareForCollectionRound( ada::ReducedNumericalModel& model )
{
  solverTask_->Restore();
  ada::ModelOperatorFacade& mof = model.GetOperator();
  mof.RefreshLocalMemory();
}

template <class T>
aaegf::GPUSolverFacade::Tuple<T>::Tuple( assg::GPUTask& bcTask, T& command, 
                                         size_type blockSize )
{
  Task = &bcTask;
  GPUCommand = &command;
  BlockSize = blockSize;
}

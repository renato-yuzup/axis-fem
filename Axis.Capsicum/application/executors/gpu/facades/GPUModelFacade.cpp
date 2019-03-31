#include "GPUModelFacade.hpp"
#include "application/executors/gpu/commands/InitBucketCommand.hpp"
#include "services/memory/gpu/EBlockLayout.hpp"
#include "domain/analyses/ModelDynamics.hpp"
#include "domain/analyses/ModelKinematics.hpp"
#include "domain/analyses/ReducedNumericalModel.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/formulations/FormulationStrategy.hpp"
#include "domain/materials/MaterialStrategy.hpp"
#include <iostream>

namespace aaegc = axis::application::executors::gpu::commands;
namespace aaegf = axis::application::executors::gpu::facades;
namespace asmg  = axis::services::memory::gpu;
namespace ada   = axis::domain::analyses;
namespace ade   = axis::domain::elements;
namespace adf   = axis::domain::formulations;
namespace adm   = axis::domain::materials;
namespace assg  = axis::services::scheduling::gpu;
namespace afm   = axis::foundation::memory;

aaegf::GPUModelFacade::GPUModelFacade( void )
{
  // nothing to do here
}

aaegf::GPUModelFacade::~GPUModelFacade( void )
{
  task_list::const_iterator end = tasks_.end();
  for (task_list::const_iterator it = tasks_.begin(); it != end; ++it)
  {
    delete (*it).Task;
  }
  tasks_.clear();
}

void aaegf::GPUModelFacade::Destroy( void ) const
{
  delete this;
}

void aaegf::GPUModelFacade::AddElementTask( assg::GPUTask& elementTask,
  ade::FiniteElement& elementSample)
{
  adf::FormulationStrategy& 
    formStrategy = elementSample.GetGPUFormulationStrategy();
  adm::MaterialStrategy& 
    matStrategy = elementSample.GetGPUMaterialStrategy();
  size_type formulationSize = elementSample.GetFormulationBlockSize();
  size_type materialSize = elementSample.GetMaterialBlockSize();
  int dofCount = elementSample.Geometry().GetTotalDofCount();
  tasks_.push_back(Tuple(elementTask, formStrategy, matStrategy, 
    formulationSize, materialSize, dofCount));  
}

void aaegf::GPUModelFacade::InitElementBuckets( void )
{
  afm::RelativePointer modelPtr = GetModelPointer();
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    Tuple& tp = *it;
    assg::GPUTask& task = *tp.Task;
    asmg::EBlockLayout memLayout(tp.FormulationBlockSize, 
      tp.MaterialBlockSize);
    size_type totalBlockSize = memLayout.GetSegmentSize();
    aaegc::InitBucketCommand initCmd(modelPtr, totalBlockSize);
    task.RunCommand(initCmd);
  }
  // must completely initialize elements before proceeding
  Synchronize();
}

void aaegf::GPUModelFacade::AllocateMemory( void )
{
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    (*it).Task->AllocateMemory();
  }
}

void aaegf::GPUModelFacade::DeallocateMemory( void )
{
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    (*it).Task->DeallocateMemory();
  }
}

void aaegf::GPUModelFacade::InitMemory( void )
{
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    (*it).Task->InitMemory();
  }
}

void aaegf::GPUModelFacade::Mirror( void )
{
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    (*it).Task->Mirror();
  }
}

void aaegf::GPUModelFacade::Restore( void )
{
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    (*it).Task->Restore();
  }
}

void aaegf::GPUModelFacade::RefreshLocalMemory( void )
{
  Restore();
}

void aaegf::GPUModelFacade::Synchronize( void )
{
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    Tuple& tp = *it;
    assg::GPUTask& task = *tp.Task;
    task.Synchronize();
  }
}

void aaegf::GPUModelFacade::CalculateGlobalLumpedMass( real t, real dt )
{
  throw std::exception("The method or operation is not implemented.");
}

void aaegf::GPUModelFacade::CalculateGlobalConsistentMass( real t, real dt )
{
  throw std::exception("The method or operation is not implemented.");
}

void aaegf::GPUModelFacade::CalculateGlobalStiffness( real t, real dt )
{
  throw std::exception("The method or operation is not implemented.");
}

void aaegf::GPUModelFacade::CalculateGlobalInternalForce( real time, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  afm::RelativePointer modelPtr = GetModelPointer();
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    Tuple& tp = *it;
    assg::GPUTask& task = *tp.Task;
    asmg::EBlockLayout memLayout(tp.FormulationBlockSize, 
      tp.MaterialBlockSize);
    size_type totalBlockSize = memLayout.GetSegmentSize();
    adf::FormulationStrategy& formStrategy = *tp.GPUFormulation;
    auto& internalForceCmd = formStrategy.GetUpdateInternalForceStrategy();
    internalForceCmd.SetParameters(totalBlockSize, modelPtr, time, 
      lastTimeIncrement, nextTimeIncrement);
    task.RunCommand(internalForceCmd);
  }
}

void aaegf::GPUModelFacade::UpdateStrain( real time, real lastTimeIncrement,
                                         real nextTimeIncrement)
{
  afm::RelativePointer modelPtr = GetModelPointer();
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    Tuple& tp = *it;
    assg::GPUTask& task = *tp.Task;
    asmg::EBlockLayout memLayout(tp.FormulationBlockSize, 
      tp.MaterialBlockSize);
    size_type totalBlockSize = memLayout.GetSegmentSize();
    auto& formStrategy = *tp.GPUFormulation;
    auto& updateCmd = formStrategy.GetUpdateStrainCommand();
    updateCmd.SetParameters(totalBlockSize, modelPtr, time, lastTimeIncrement, 
      nextTimeIncrement);
    task.RunCommand(updateCmd);
  }
}

void aaegf::GPUModelFacade::UpdateStress( real time, real lastTimeIncrement, 
                                         real nextTimeIncrement)
{
  afm::RelativePointer modelPtr = GetModelPointer();
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    Tuple& tp = *it;
    assg::GPUTask& task = *tp.Task;
    asmg::EBlockLayout memLayout(tp.FormulationBlockSize, 
      tp.MaterialBlockSize);
    size_type totalBlockSize = memLayout.GetSegmentSize();
    adm::MaterialStrategy& materialStrategy = *tp.GPUMaterial;
    adm::UpdateStressCommand& stressCmd = 
      materialStrategy.GetUpdateStressCommand();
    stressCmd.SetParameters(totalBlockSize, modelPtr, time, 
      lastTimeIncrement, nextTimeIncrement);
    task.RunCommand(stressCmd);
  }
}

void aaegf::GPUModelFacade::UpdateGeometry( real time, real lastTimeIncrement,
  real nextTimeIncrement )
{
  afm::RelativePointer modelPtr = GetModelPointer();
  task_list::iterator end = tasks_.end();
  for (task_list::iterator it = tasks_.begin(); it != end; ++it)
  {
    Tuple& tp = *it;
    assg::GPUTask& task = *tp.Task;
    asmg::EBlockLayout memLayout(tp.FormulationBlockSize, 
      tp.MaterialBlockSize);
    size_type totalBlockSize = memLayout.GetSegmentSize();
    auto& formStrategy = *tp.GPUFormulation;
    auto& updateCmd = formStrategy.GetUpdateGeometryCommand();
    updateCmd.SetParameters(totalBlockSize, modelPtr, time, lastTimeIncrement, 
      nextTimeIncrement);
    task.RunCommand(updateCmd);
  }
}

void aaegf::GPUModelFacade::UpdateNodeQuantities( void )
{
  throw std::exception("The method or operation is not implemented.");
}


/***************************** TUPLE MEMBERS **********************************/
aaegf::GPUModelFacade::Tuple::Tuple( assg::GPUTask& elementTask, 
  adf::FormulationStrategy&formulation, adm::MaterialStrategy &material,
  size_type formulationBlockSize, size_type materialBlockSize, int dofCount) :
GPUFormulation(&formulation), GPUMaterial(&material), Task(&elementTask)
{
  FormulationBlockSize = formulationBlockSize;
  MaterialBlockSize = materialBlockSize;
  TotalDofCount = dofCount;
}

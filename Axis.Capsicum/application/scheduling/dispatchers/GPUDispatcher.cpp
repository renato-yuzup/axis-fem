#include "GPUDispatcher.hpp"
#include "DataCoalescer.hpp"
#include "GPUErrorHandler.hpp"
#include "GPUFiniteElementAnalysis.hpp"
#include "application/executors/gpu/commands/FiniteElementInitCommand.hpp"
#include "application/executors/gpu/commands/SolverInitializerCommand.hpp"
#include "services/memory/gpu/EBlockLayout.hpp"
#include "domain/algorithms/Solver.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/analyses/ReducedNumericalModel.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "services/memory/NoMemoryLayout.hpp"
#include "services/memory/commands/MemoryCommand.hpp"
#include "services/memory/commands/NullMemoryCommand.hpp"
#include "services/scheduling/gpu/KernelScheduler.hpp"
#include "foundation/memory/pointer.hpp"
#include "services/memory/gpu/CBlockLayout.hpp"
#include "application/executors/gpu/commands/CurveInitCommand.hpp"
#include "services/memory/gpu/BcBlockLayout.hpp"
#include "application/executors/gpu/commands/BoundaryConditionInitCommand.hpp"
#include <iostream>

namespace aass  = axis::application::scheduling::scheduler;
namespace aaegc = axis::application::executors::gpu::commands;
namespace aaegf = axis::application::executors::gpu::facades;
namespace aasd  = axis::application::scheduling::dispatchers;
namespace asmc  = axis::services::memory::commands;
namespace asmo  = axis::services::memory;
namespace asmg  = axis::services::memory::gpu;
namespace assg  = axis::services::scheduling::gpu;
namespace ada   = axis::domain::analyses;
namespace adal  = axis::domain::algorithms;
namespace ada   = axis::domain::analyses;
namespace adbc  = axis::domain::boundary_conditions;
namespace adc   = axis::domain::collections;
namespace adcu  = axis::domain::curves;
namespace ade   = axis::domain::elements;
namespace afc   = axis::foundation::computing;
namespace afm   = axis::foundation::memory;

namespace {
void ScheduleBoundaryCondition(aaegf::GPUSolverFacade& solverFacade, 
  axis::domain::boundary_conditions::BoundaryCondition::ConstraintType bcType,
  aasd::DataCoalescer& coalescer, assg::KernelScheduler& scheduler,
  assg::GPUSink& errSink)
{
  scheduler.BeginGPURound();
  int accBcGroupCount = coalescer.GetBcGroupCount(bcType);
  for (int i = 0; i < accBcGroupCount; ++i)
  {
    size_type groupSize = coalescer.GetBcGroupSize(bcType, i);
    void **bcs = new void *[groupSize];
    coalescer.FillBcGroupArray(bcType, bcs, i);

    // sample a curve in the group
    adbc::BoundaryCondition& bc = *(adbc::BoundaryCondition *)bcs[0];
    asmg::BcBlockLayout blockLayout(bc.GetGPUDataSize());
    aaegc::BoundaryConditionInitCommand bcInitCmd(bcs, groupSize);
    assg::GPUTask *task = scheduler.ScheduleLocal(bcInitCmd, blockLayout,
      groupSize, errSink);
    solverFacade.AddBoundaryConditionTask(bcType, *task, bc.GetUpdateCommand(), 
      bc.GetGPUDataSize());
    delete [] bcs;
  }
  scheduler.EndGPURound();
} 

void ScheduleCurve(aaegf::GPUSolverFacade& solverFacade, 
  aasd::DataCoalescer& coalescer, assg::KernelScheduler& scheduler,
  assg::GPUSink& errSink)
{
  // schedule curves
  scheduler.BeginGPURound();
  int curveGroupCount = coalescer.GetCurveGroupCount();
  for (int i = 0; i < curveGroupCount; ++i)
  {
    size_type groupSize = coalescer.GetCurveGroupSize(i);
    void **curves = new void *[groupSize];
    coalescer.FillCurveGroupArray(curves, i);

    // sample a curve in the group
    adcu::Curve& c = *(adcu::Curve *)curves[0];
    asmg::CBlockLayout blockLayout(c.GetGPUDataSize());
    aaegc::CurveInitCommand curveInitCmd(curves, groupSize);
    assg::GPUTask *task = scheduler.ScheduleLocal(curveInitCmd, blockLayout,
      groupSize, errSink);
    solverFacade.AddCurveTask(*task, c.GetUpdateCommand(), c.GetGPUDataSize());
    delete [] curves;
  }
  scheduler.EndGPURound();
}
} // namespace

aasd::GPUDispatcher::GPUDispatcher( void )
{
  // nothing to do here
}

aasd::GPUDispatcher::~GPUDispatcher( void )
{
  // nothing to do here
}

void aasd::GPUDispatcher::DispatchJob( aass::ExecutionRequest& jobRequest, 
  afc::ResourceManager& manager )
{
  adal::Solver& solver = jobRequest.GetSolver();
  ada::NumericalModel& model = jobRequest.GetNumericalModel();
  ada::AnalysisTimeline& timeline = jobRequest.GetTimeline();

  DataCoalescer coalescer;
  aaegf::GPUModelFacade modelFacade;
  aaegf::GPUSolverFacade solverFacade;
  assg::KernelScheduler scheduler(manager);
  GPUErrorHandler errSink;

  // group similar data
  coalescer.Coalesce(model);

  // prepare numerical model and solver to run in GPU
  PrepareNumericalModel(model);
  solver.AllocateGPUData(model, timeline);  
  afm::RelativePointer reducedModelPtr = 
    ada::ReducedNumericalModel::Create(model, modelFacade);
  modelFacade.SetTargetModel(reducedModelPtr);
  
  // after everything has been initialized and assuming no more model memory 
  // allocations will occur (except those pertaining to the solver itself), 
  // init solver facade
  SetUpSolverFacade(solverFacade, coalescer, scheduler, solver, model, 
    timeline, errSink);

  // initialize model facade
  BuildModelFacade(modelFacade, coalescer, scheduler, errSink);

  // build analysis and run
  GPUFiniteElementAnalysis analysis(solver, solverFacade, reducedModelPtr, 
    modelFacade);
  analysis.ConnectListener(*this);
  analysis.Run(timeline);
  analysis.DisconnectListener(*this);
}

void aasd::GPUDispatcher::BuildModelFacade( aaegf::GPUModelFacade& facade, 
  aasd::DataCoalescer& coalescer, assg::KernelScheduler& scheduler, 
  assg::GPUSink& errSink )
{
  scheduler.BeginGPURound();
  for (size_type i = 0; i < coalescer.GetElementGroupCount(); i++)
  {
    size_type groupSize = coalescer.GetElementGroupSize(i);
    void **ptrArray = new void *[groupSize];
    coalescer.FillElementGroupArray(ptrArray, i);
    ade::FiniteElement& fe = *(ade::FiniteElement *)(ptrArray[0]);
    asmg::EBlockLayout blockLayout(fe.GetFormulationBlockSize(), 
      fe.GetMaterialBlockSize());
    aaegc::FiniteElementInitCommand initMemCmd(ptrArray, groupSize);
    assg::GPUTask *task = scheduler.ScheduleLocal(
      initMemCmd,  // command object which dispatches memory initialization
      blockLayout, // how each memory block is organized
      groupSize,   // how many elements will be scheduled among installed GPUs
      errSink);    // object that receives any error occurred
    facade.AddElementTask(*task, fe);
    delete [] ptrArray;
  }
  scheduler.EndGPURound();
}

void aasd::GPUDispatcher::PrepareNumericalModel( ada::NumericalModel& model )
{
  size_type nodeCount = model.Nodes().Count();
  for (size_type i = 0; i < nodeCount; ++i)
  {
    ade::Node& node = model.Nodes().GetByPosition(i);
    node.CompileConnectivityList();
  }
}

void aasd::GPUDispatcher::SetUpSolverFacade(aaegf::GPUSolverFacade& solverFacade, 
  DataCoalescer& coalescer, assg::KernelScheduler& scheduler, 
  adal::Solver& solver, ada::NumericalModel& model, 
  ada::AnalysisTimeline& timeline, assg::GPUSink& errSink)
{
  ScheduleCurve(solverFacade, coalescer, scheduler, errSink);
  ScheduleBoundaryCondition(solverFacade, 
    adbc::BoundaryCondition::PrescribedAcceleration, coalescer, scheduler, 
    errSink);
  ScheduleBoundaryCondition(solverFacade, 
    adbc::BoundaryCondition::PrescribedVelocity, coalescer, scheduler, 
    errSink);
  ScheduleBoundaryCondition(solverFacade, 
    adbc::BoundaryCondition::PrescribedDisplacement, coalescer, scheduler, 
    errSink);
  ScheduleBoundaryCondition(solverFacade, 
    adbc::BoundaryCondition::NodalLoad, coalescer, scheduler, 
    errSink);
  ScheduleBoundaryCondition(solverFacade, 
    adbc::BoundaryCondition::Lock, coalescer, scheduler, 
    errSink);

  // create solver task
  assg::GPUTask *solverTask = scheduler.ScheduleGlobal(
    solver.GetGPUThreadsRequired(model), 
    aaegc::SolverInitializerCommand(solver, model, timeline), errSink);
  solverFacade.SetSolverTask(*solverTask);
}

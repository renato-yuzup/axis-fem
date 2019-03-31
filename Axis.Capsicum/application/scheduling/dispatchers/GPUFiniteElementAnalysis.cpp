#include "GPUFiniteElementAnalysis.hpp"
#include <mutex>
#include "domain/analyses/AnalysisTimeline.hpp"
#include "domain/algorithms/messages/SnapshotEndMessage.hpp"
#include "system_messages.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include <iostream>

namespace aasd  = axis::application::scheduling::dispatchers;
namespace aaegf = axis::application::executors::gpu::facades;
namespace ada   = axis::domain::analyses;
namespace adalm = axis::domain::algorithms::messages;
namespace adal  = axis::domain::algorithms;
namespace afm   = axis::foundation::memory;
namespace ayfm  = axis::yuzu::foundation::memory;

aasd::GPUFiniteElementAnalysis::GPUFiniteElementAnalysis(
  adal::Solver& gpuSolver, aaegf::GPUSolverFacade& solverFacade,
  afm::RelativePointer& gpuNumericalModel, aaegf::GPUModelFacade& modelFacade) :
gpuSolver_(gpuSolver), solverFacade_(solverFacade), 
gpuNumericalModel_(gpuNumericalModel), modelFacade_(modelFacade)
{
  // nothing to do here
}

aasd::GPUFiniteElementAnalysis::~GPUFiniteElementAnalysis(void)
{
  // nothing to do here
}

void aasd::GPUFiniteElementAnalysis::Run( ada::AnalysisTimeline& timeline )
{
  // register memory for use in GPU via DMA
  solverFacade_.ClaimMemory(&timeline, sizeof(ada::AnalysisTimeline));
  solverFacade_.AllocateMemory();
  modelFacade_.AllocateMemory();
  solverFacade_.InitMemory();
  modelFacade_.InitMemory();

  // notify all plugins that now we have a model arena in GPU correctly set up
  void *gpuModelArenaAddr = solverFacade_.GetGPUModelArenaAddress();
  System::Broadcast(AXIS_SYS_GPU_MEMORY_ARENA_INIT, gpuModelArenaAddr);
  // since we are not subscribed in system hook, we need to change model arena
  // address manually
  ayfm::SetGPUArena(gpuModelArenaAddr);

  solverFacade_.Mirror();
  modelFacade_.Mirror();
  modelFacade_.InitElementBuckets();

  solverFacade_.ConnectListener(*this);
  gpuSolver_.RunOnGPU(timeline, gpuNumericalModel_, solverFacade_);
  solverFacade_.DisconnectListener(*this);

  modelFacade_.Restore();
  solverFacade_.Restore();
  modelFacade_.DeallocateMemory();
  solverFacade_.DeallocateMemory();
  solverFacade_.ReturnMemory(&timeline);
}

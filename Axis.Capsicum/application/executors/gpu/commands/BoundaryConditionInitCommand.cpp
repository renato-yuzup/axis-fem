#include "BoundaryConditionInitCommand.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"
#include "services/memory/gpu/BcBlockLayout.hpp"
#include "foundation/memory/pointer.hpp"

namespace aaegc = axis::application::executors::gpu::commands;
namespace asmg  = axis::services::memory::gpu;
namespace adbc  = axis::domain::boundary_conditions;
namespace afm   = axis::foundation::memory;
namespace asmc  = axis::services::memory::commands;

aaegc::BoundaryConditionInitCommand::BoundaryConditionInitCommand( 
  void **associatedDataPtr, uint64 bcCount ) :
ElementInitializerCommand(associatedDataPtr, bcCount)
{
  // nothing to do here
}

aaegc::BoundaryConditionInitCommand::~BoundaryConditionInitCommand( void )
{
  // nothing to do here
}

asmc::ElementInitializerCommand& aaegc::BoundaryConditionInitCommand::DoSlice( 
  void **associatedDataPtr, uint64 elementCount ) const
{
  return *new BoundaryConditionInitCommand(associatedDataPtr,   
    elementCount);
}

void aaegc::BoundaryConditionInitCommand::DoExecute( void *blockDataAddress,
  const void *externalMirroredAddress, void *associatedData )
{
  adbc::BoundaryCondition& bc = *(adbc::BoundaryCondition *)associatedData;
  int bcBlockSize = bc.GetGPUDataSize();
  asmg::BcBlockLayout cpuLayout(bcBlockSize);
  uint64 dofId = bc.GetDoF()->GetId();
  cpuLayout.InitMemoryBlock(blockDataAddress, dofId);
  bc.InitGPUData(cpuLayout.GetCustomDataAddress(blockDataAddress),
    *cpuLayout.GetOutputBucketAddress(blockDataAddress));
}

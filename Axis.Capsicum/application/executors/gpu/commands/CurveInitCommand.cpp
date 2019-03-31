#include "CurveInitCommand.hpp"
#include "domain/curves/Curve.hpp"
#include "services/memory/gpu/CBlockLayout.hpp"
#include "foundation/memory/pointer.hpp"

namespace aaegc = axis::application::executors::gpu::commands;
namespace asmg  = axis::services::memory::gpu;
namespace adcu  = axis::domain::curves;
namespace afm   = axis::foundation::memory;
namespace asmc  = axis::services::memory::commands;

aaegc::CurveInitCommand::CurveInitCommand( 
  void **associatedDataPtr, uint64 curveCount ) :
ElementInitializerCommand(associatedDataPtr, curveCount)
{
  // nothing to do here
}

aaegc::CurveInitCommand::~CurveInitCommand( void )
{
  // nothing to do here
}

asmc::ElementInitializerCommand& aaegc::CurveInitCommand::DoSlice( 
  void **associatedDataPtr, uint64 elementCount ) const
{
  return *new CurveInitCommand(associatedDataPtr, elementCount);
}

void aaegc::CurveInitCommand::DoExecute( void *blockDataAddress,
  const void *externalMirroredAddress, void *associatedData)
{
  adcu::Curve& curve = *(adcu::Curve *)associatedData;
  int curveBlockSize = curve.GetGPUDataSize();
  asmg::CBlockLayout cpuLayout(curveBlockSize);
  asmg::CBlockLayout gpuLayout(curveBlockSize);
  cpuLayout.InitMemoryBlock(blockDataAddress);
  curve.InitGPUData(cpuLayout.GetCustomDataAddress(blockDataAddress),
    cpuLayout.GetOutputBucketAddress(blockDataAddress),
    gpuLayout.GetOutputBucketAddress(externalMirroredAddress));
}

#include "NullKernelCommand.hpp"

namespace afc = axis::foundation::computing;

afc::NullKernelCommand::NullKernelCommand(void)
{
  // nothing to do here
}

afc::NullKernelCommand::~NullKernelCommand(void)
{
  // nothing to do here
}

void afc::NullKernelCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  // do nothing
}

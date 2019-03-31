#include "MultiLineCurveCommand.hpp"
#include "multi_line_curve_kernel.hpp"

namespace adcu = axis::domain::curves;

adcu::MultiLineCurveCommand::MultiLineCurveCommand(void)
{
  // nothing to do here
}


adcu::MultiLineCurveCommand::~MultiLineCurveCommand(void)
{
  // nothing to do here
}

void adcu::MultiLineCurveCommand::Run( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  UpdateCurveOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, GetTime());
}
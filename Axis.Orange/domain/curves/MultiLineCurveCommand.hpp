#pragma once
#include "domain/curves/CurveUpdateCommand.hpp"

namespace axis { namespace domain { namespace curves {

class MultiLineCurveCommand : public CurveUpdateCommand
{
public:
  MultiLineCurveCommand(void);
  ~MultiLineCurveCommand(void);
  virtual void Run( uint64 numThreadsToUse, uint64 startIndex, 
    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
    const axis::Dimension3D& blockDim, void * streamPtr );
};

} } } // namespace axis::domain::curves

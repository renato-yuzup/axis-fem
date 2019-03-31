#include "CurveData.hpp"
#define MEMORY_WORD_LENGTH 16
#define ALIGN_ADDRESS(x)   ((x) + MEMORY_WORD_LENGTH - 1) & ~(MEMORY_WORD_LENGTH - 1)

namespace aydcu = axis::yuzu::domain::curves;

GPU_ONLY aydcu::CurveData::CurveData( void *baseAddress, uint64 index, 
                                      int curveDataSize )
{
  uint64 baseSize = ALIGN_ADDRESS(sizeof(real));
  uint64 extendedSize = ALIGN_ADDRESS(curveDataSize);
  uint64 totalSize = baseSize + extendedSize;
  blockAddress_ = (void *)((uint64)baseAddress + index*totalSize);
}

GPU_ONLY aydcu::CurveData::~CurveData( void )
{
  // nothing to do here
}

GPU_ONLY real * aydcu::CurveData::GetOutputBucket( void ) const
{
  return (real *)blockAddress_;
}

GPU_ONLY void * aydcu::CurveData::GetCurveData( void ) const
{
  return (void *)((uint64)blockAddress_ + ALIGN_ADDRESS(sizeof(real)));
}

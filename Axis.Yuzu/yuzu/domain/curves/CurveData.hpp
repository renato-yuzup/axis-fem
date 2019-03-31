#pragma once
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu { namespace domain { namespace curves {

class CurveData
{
public:
  GPU_ONLY CurveData(void *baseAddress, uint64 index, int curveDataSize);
  GPU_ONLY ~CurveData(void);
  GPU_ONLY real *GetOutputBucket(void) const;
  GPU_ONLY void *GetCurveData(void) const;
private:
  void *blockAddress_;
};

} } } } // namespace axis::yuzu::domain::curves

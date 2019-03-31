#pragma once
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu { namespace domain { namespace boundary_conditions {

class BoundaryConditionData
{
public:
  GPU_ONLY BoundaryConditionData(void *bcDataAddress, uint64 index, int bcDataSize);
  GPU_ONLY ~BoundaryConditionData(void);
  GPU_ONLY real *GetOutputBucket(void) const;
  GPU_ONLY uint64 GetDofId(void) const;
  GPU_ONLY void *GetCustomData(void) const;
private:
  void *startingAddress_;
};

} } } } // namespace axis::yuzu::domain::boundary_conditions

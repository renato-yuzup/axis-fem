#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/domain/analyses/ModelDynamics.hpp"
#include "yuzu/domain/analyses/ModelKinematics.hpp"
#include "yuzu/domain/elements/FiniteElement.hpp"
#include "yuzu/domain/elements/Node.hpp"

namespace axis { namespace yuzu { namespace domain { namespace analyses {

/**
 * Implements a version of the numerical model with reduced functionality,
 * so that it can be used with external processing devices.
**/
class ReducedNumericalModel
{
public:
  GPU_ONLY ~ReducedNumericalModel(void);
  GPU_ONLY const ModelDynamics& Dynamics(void) const;
  GPU_ONLY ModelDynamics& Dynamics(void);
  GPU_ONLY const ModelKinematics& Kinematics(void) const;
  GPU_ONLY ModelKinematics& Kinematics(void);
  GPU_ONLY size_type GetElementCount(void) const;
  GPU_ONLY size_type GetNodeCount(void) const;
  GPU_ONLY const axis::yuzu::domain::elements::FiniteElement& GetElement(size_type index) const;
  GPU_ONLY axis::yuzu::domain::elements::FiniteElement& GetElement(size_type index);
  GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetElementPointer(size_type index) const;
  GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetElementPointer(size_type index);
  GPU_ONLY const axis::yuzu::domain::elements::Node& GetNode(size_type index) const;
  GPU_ONLY axis::yuzu::domain::elements::Node& GetNode(size_type index);
  GPU_ONLY const axis::yuzu::foundation::memory::RelativePointer GetNodePointer(size_type index) const;
  GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetNodePointer(size_type index);
  GPU_ONLY const real *GetElementOutputBucket(size_type eIdx) const;
  GPU_ONLY void SetElementOutputBucket(size_type eIdx, real *bucket);
private:
  GPU_ONLY ReducedNumericalModel(void);
  void *operator_; // reserved
  axis::yuzu::foundation::memory::RelativePointer nodeArrayPtr_;
  axis::yuzu::foundation::memory::RelativePointer elementArrayPtr_;
  axis::yuzu::foundation::memory::RelativePointer outputBucketArrayPtr_;
  size_type elementCount_;
  size_type nodeCount_;
  axis::yuzu::foundation::memory::RelativePointer kinematics_;
  axis::yuzu::foundation::memory::RelativePointer dynamics_;
};

} } } } // namespace axis::yuzu::domain::analyses

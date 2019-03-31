#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"

namespace axis { namespace yuzu { namespace domain { namespace physics {

class InfinitesimalState;

/**
 * Adapter class that protects physical state of an infinitesimal so that
 * only adequate properties are modified.
**/
class UpdatedPhysicalState
{
public:
  GPU_ONLY UpdatedPhysicalState(InfinitesimalState& state);
  GPU_ONLY ~UpdatedPhysicalState(void);
  GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Stress(void);
  GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Stress(void) const;
  GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& LastStressIncrement(void);
  GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& LastStressIncrement(void) const;
  GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& PlasticStrain(void);
  GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& PlasticStrain(void) const;
  GPU_ONLY real& EffectivePlasticStrain(void);
  GPU_ONLY real EffectivePlasticStrain(void) const;
private:
  InfinitesimalState& state_;
};

} } } } // namespace axis::yuzu::domain::physics

#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/fwd/finite_element_fwd.hpp"
#include "foundation/blas/blas.hpp"

namespace axis { namespace domain { namespace physics {

/**
 * Adapter class that protects physical state of an infinitesimal so that
 * only adequate properties are modified.
**/
class AXISCOMMONLIBRARY_API UpdatedPhysicalState
{
public:
  UpdatedPhysicalState(InfinitesimalState& state);
  ~UpdatedPhysicalState(void);

  axis::foundation::blas::ColumnVector& Stress(void);
  const axis::foundation::blas::ColumnVector& Stress(void) const;

  axis::foundation::blas::ColumnVector& LastStressIncrement(void);
  const axis::foundation::blas::ColumnVector& LastStressIncrement(void) const;

  axis::foundation::blas::ColumnVector& PlasticStrain(void);
  const axis::foundation::blas::ColumnVector& PlasticStrain(void) const;

  real& EffectivePlasticStrain(void);
  real EffectivePlasticStrain(void) const;
private:
  InfinitesimalState& state_;
};

} } } // namespace axis::domain::physics

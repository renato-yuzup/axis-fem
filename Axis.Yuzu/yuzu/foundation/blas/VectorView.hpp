#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**
 * Provides a relation between a vector and a subvector composed of its elements.
**/
class VectorView
{
public:
  GPU_ONLY VectorView(size_type size);
  GPU_ONLY ~VectorView(void);				
  GPU_ONLY size_type GetMaskIndex(size_type index) const;
  GPU_ONLY void SetMaskIndex(size_type index, size_type maskIndex) const;
  GPU_ONLY size_type operator ()(size_type index) const;
  GPU_ONLY size_type Length(void) const;
private:
  size_type maskRelativePos_;
  size_type maskLength_;
};

} } } } // namespace axis::yuzu::foundation::blas

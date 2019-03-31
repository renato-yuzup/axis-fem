#pragma once
#include "foundation/blas/Axis.BLAS.hpp"
#include "foundation/axis.SystemBase.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace foundation { namespace blas {

/**
 * Provides a relation between a vector and a subvector composed of its elements.
**/
class AXISBLAS_API VectorView
{
public:
	~VectorView(void);				
	size_type GetMaskIndex(size_type index) const;
	void SetMaskIndex(size_type index, size_type maskIndex) const;
	size_type operator ()(size_type index) const;
	size_type Length(void) const;
#if !defined(__CUDA_ARCH__)
  static axis::foundation::memory::RelativePointer Create(size_type size);
#endif
private:
  VectorView(size_type size);
  void *operator new(size_t, void *ptr);
  void operator delete(void *, void *);

  size_type maskRelativePos_;
  size_type maskLength_;
};

} } } // namespace axis::foundation::blas

#pragma once
#include "foundation/blas/Axis.BLAS.hpp"
#include "foundation/axis.SystemBase.hpp"
#include "foundation/blas/RowVector.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace foundation { namespace blas {

class AXISBLAS_API SubRowVector
{
public:
  typedef RowVector    inner_type;
  typedef SubRowVector self;

	~SubRowVector(void);
	real GetElement( size_type pos ) const;
	self& SetElement( size_type pos, real value );
	self& Accumulate( size_type pos, real value );
	self& CopyFrom( const inner_type& vector );
  self& CopyFrom( const self& vector );
	self& ClearAll( void );
	self& Scale( real value );
	self& SetAll( real value );
	size_type Length( void ) const;
	real SelfScalarProduct( void ) const;
	real Norm( void ) const;
	self& Invert( void );
	size_type Rows( void ) const;
	size_type Columns( void ) const;
	real operator ()(size_type pos) const;
	real& operator ()(size_type pos);
#if !defined(__CUDA_ARCH__)
  static axis::foundation::memory::RelativePointer Create(
            axis::foundation::memory::RelativePointer& targetVector, 
            const axis::foundation::memory::RelativePointer& mask);
#endif
private:  
	SubRowVector(axis::foundation::memory::RelativePointer& targetVector, 
                      const axis::foundation::memory::RelativePointer& mask);
  void *operator new(size_t, void *ptr);
  void operator delete(void *, void *);

  axis::foundation::memory::RelativePointer mask_;
  axis::foundation::memory::RelativePointer targetVector_;
};

} } } // namespace axis::foundation::blas


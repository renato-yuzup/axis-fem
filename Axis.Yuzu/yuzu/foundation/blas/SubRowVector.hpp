#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/blas/RowVector.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

class SubRowVector
{
public:
  typedef RowVector    inner_type;
  typedef SubRowVector self;

  GPU_ONLY SubRowVector(axis::yuzu::foundation::memory::RelativePointer& targetVector, 
                        const axis::yuzu::foundation::memory::RelativePointer& mask);
	GPU_ONLY ~SubRowVector(void);
	GPU_ONLY real GetElement( size_type pos ) const;
	GPU_ONLY self& SetElement( size_type pos, real value );
	GPU_ONLY self& Accumulate( size_type pos, real value );
	GPU_ONLY self& CopyFrom( const inner_type& vector );
  GPU_ONLY self& CopyFrom( const self& vector );
	GPU_ONLY self& ClearAll( void );
	GPU_ONLY self& Scale( real value );
	GPU_ONLY self& SetAll( real value );
	GPU_ONLY size_type Length( void ) const;
	GPU_ONLY real SelfScalarProduct( void ) const;
	GPU_ONLY real Norm( void ) const;
	GPU_ONLY self& Invert( void );
	GPU_ONLY size_type Rows( void ) const;
	GPU_ONLY size_type Columns( void ) const;
	GPU_ONLY real operator ()(size_type pos) const;
	GPU_ONLY real& operator ()(size_type pos);
private:  
  axis::yuzu::foundation::memory::RelativePointer mask_;
  axis::yuzu::foundation::memory::RelativePointer targetVector_;
};

} } } } // namespace axis::yuzu::foundation::blas


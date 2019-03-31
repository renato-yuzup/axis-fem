#include "VectorView.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayfb::VectorView::VectorView( size_type size )
{
	maskRelativePos_ = sizeof(VectorView);
	maskLength_ = size;
}

GPU_ONLY ayfb::VectorView::~VectorView( void )
{
  // nothing to do here
}

GPU_ONLY size_type ayfb::VectorView::GetMaskIndex( size_type index ) const
{
	return ((size_type *)((uint64)this + maskRelativePos_))[index];
}

GPU_ONLY size_type ayfb::VectorView::operator()( size_type index ) const
{
  return ((size_type *)((uint64)this + maskRelativePos_))[index];
}

GPU_ONLY void ayfb::VectorView::SetMaskIndex( size_type index, size_type maskIndex ) const
{
  ((size_type *)((uint64)this + maskRelativePos_))[index] = maskIndex;
}

GPU_ONLY size_type ayfb::VectorView::Length( void ) const
{
	return maskLength_;
}

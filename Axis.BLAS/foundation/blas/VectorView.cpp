#include "VectorView.hpp"
#include "foundation/memory/pointer.hpp"
#include "foundation/OutOfBoundsException.hpp"

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

afb::VectorView::VectorView( size_type size )
{
	maskRelativePos_ = sizeof(VectorView);
	maskLength_ = size;
}

afb::VectorView::~VectorView( void )
{
  // nothing to do here
}

size_type afb::VectorView::GetMaskIndex( size_type index ) const
{
	if (index >= maskLength_)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	return ((size_type *)((uint64)this + maskRelativePos_))[index];
}

size_type afb::VectorView::operator()( size_type index ) const
{
	if (index >= maskLength_)
	{
		throw axis::foundation::OutOfBoundsException();
	}
  return ((size_type *)((uint64)this + maskRelativePos_))[index];
}

void afb::VectorView::SetMaskIndex( size_type index, size_type maskIndex ) const
{
	if (index >= maskLength_)
	{
		throw axis::foundation::OutOfBoundsException();
	}
  ((size_type *)((uint64)this + maskRelativePos_))[index] = maskIndex;
}

size_type afb::VectorView::Length( void ) const
{
	return maskLength_;
}

afm::RelativePointer afb::VectorView::Create( size_type size )
{
  afm::RelativePointer ptr = 
    System::ModelMemory().Allocate(sizeof(afb::VectorView) + sizeof(size_type)*size);
  new (*ptr) VectorView(size);
  return ptr;
}

void * axis::foundation::blas::VectorView::operator new( size_t, void *ptr )
{
  return ptr;
}

void axis::foundation::blas::VectorView::operator delete( void *, void * )
{
  // nothing to do here
}

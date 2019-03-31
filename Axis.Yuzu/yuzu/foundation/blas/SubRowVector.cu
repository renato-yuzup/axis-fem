#include "SubRowVector.hpp"
#include <math.h>
#include "yuzu/foundation/blas/VectorView.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayfb::SubRowVector::SubRowVector( ayfm::RelativePointer& targetVector, 
                                           const ayfm::RelativePointer& mask ) :
targetVector_(targetVector), mask_(mask)
{
  // nothing to do here
}

GPU_ONLY ayfb::SubRowVector::~SubRowVector( void )
{
	// nothing to do here
}

GPU_ONLY real ayfb::SubRowVector::GetElement( size_type pos ) const
{
	return yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos));
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::SetElement( size_type pos, real value )
{
  yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos)) = value;
	return *this;
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::Accumulate( size_type pos, real value )
{
  yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos)) += value;
	return *this;
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::CopyFrom( const self& vector )
{
	size_type len = Length();
	for (size_type i = 0; i < len; i++)
	{
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = vector(i);
	}
	return *this;
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::CopyFrom( const inner_type& vector )
{
  size_type len = Length();
  for (size_type i = 0; i < len; i++)
  {
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = vector(i);
  }
  return *this;
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::ClearAll( void )
{
	SetAll(0);
	return *this;
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::Scale( real value )
{
	size_type len = Length();
	for (size_type i = 0; i < len; i++)
	{
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) *= value;
	}
	return *this;
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::SetAll( real value )
{
	size_type len = Length();
	for (size_type i = 0; i < len; i++)
	{
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = value;
	}
	return *this;
}

GPU_ONLY size_type ayfb::SubRowVector::Length( void ) const
{
  return yabsref<ayfb::VectorView>(mask_).Length();
}

GPU_ONLY real ayfb::SubRowVector::SelfScalarProduct( void ) const
{
	size_type len = Length();
	real result = 0;
	for (size_type i = 0; i < len; i++)
	{
    real val = yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i));
		result += val*val;
	}
	return result;
}

GPU_ONLY real ayfb::SubRowVector::Norm( void ) const
{
	return sqrt(SelfScalarProduct());
}

GPU_ONLY ayfb::SubRowVector::self& ayfb::SubRowVector::Invert( void )
{
	size_type len = Length();
	for (size_type i = 0; i < len; i++)
	{
    real val = yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i));
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = 1.0 / val;
	}
	return *this;
}

GPU_ONLY size_type ayfb::SubRowVector::Rows( void ) const
{
	return 1;
}

GPU_ONLY size_type ayfb::SubRowVector::Columns( void ) const
{
	return yabsref<ayfb::VectorView>(mask_).Length();
}

GPU_ONLY real ayfb::SubRowVector::operator()( size_type pos ) const
{
  return yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos));
}

GPU_ONLY real& ayfb::SubRowVector::operator()( size_type pos )
{
  return yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos));
}

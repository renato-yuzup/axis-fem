#include "SubRowVector.hpp"
#include <math.h>
#include "foundation/blas/VectorView.hpp"
#include "foundation/memory/pointer.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/DimensionMismatchException.hpp"

#define SCHEDULE_TINY_OPS		schedule(dynamic,512)

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

afb::SubRowVector::SubRowVector( afm::RelativePointer& targetVector, const afm::RelativePointer& mask ) :
targetVector_(targetVector), mask_(mask)
{
  // nothing to do here
}

afb::SubRowVector::~SubRowVector( void )
{
	// nothing to do here
}

real afb::SubRowVector::GetElement( size_type pos ) const
{
	return absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos));
}

afb::SubRowVector::self& afb::SubRowVector::SetElement( size_type pos, real value )
{
  absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos)) = value;
	return *this;
}

afb::SubRowVector::self& afb::SubRowVector::Accumulate( size_type pos, real value )
{
  absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos)) += value;
	return *this;
}

afb::SubRowVector::self& afb::SubRowVector::CopyFrom( const self& vector )
{
	if (vector.Length() != Length())
	{
		throw axis::foundation::DimensionMismatchException();
	}
	size_type len = Length();

	#pragma omp parallel for SCHEDULE_TINY_OPS
	for (size_type i = 0; i < len; i++)
	{
    absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i)) = vector(i);
	}
	return *this;
}

afb::SubRowVector::self& afb::SubRowVector::CopyFrom( const inner_type& vector )
{
  if (vector.Length() != Length())
  {
    throw axis::foundation::DimensionMismatchException();
  }
  size_type len = Length();

  #pragma omp parallel for SCHEDULE_TINY_OPS
  for (size_type i = 0; i < len; i++)
  {
    absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i)) = vector(i);
  }
  return *this;
}

afb::SubRowVector::self& afb::SubRowVector::ClearAll( void )
{
	SetAll(0);
	return *this;
}

afb::SubRowVector::self& afb::SubRowVector::Scale( real value )
{
	size_type len = Length();

	#pragma omp parallel for SCHEDULE_TINY_OPS
	for (size_type i = 0; i < len; i++)
	{
    absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i)) *= value;
	}
	return *this;
}

afb::SubRowVector::self& afb::SubRowVector::SetAll( real value )
{
	size_type len = Length();

	#pragma omp parallel for SCHEDULE_TINY_OPS
	for (size_type i = 0; i < len; i++)
	{
    absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i)) = value;
	}
	return *this;
}

size_type afb::SubRowVector::Length( void ) const
{
  return absref<afb::VectorView>(mask_).Length();
}

real afb::SubRowVector::SelfScalarProduct( void ) const
{
	size_type len = Length();
	real result = 0;
	#pragma omp parallel for reduction(+:result) SCHEDULE_TINY_OPS
	for (size_type i = 0; i < len; i++)
	{
    real val = absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i));
		result += val*val;
	}
	return result;
}

real afb::SubRowVector::Norm( void ) const
{
	return sqrt(SelfScalarProduct());
}

afb::SubRowVector::self& afb::SubRowVector::Invert( void )
{
	size_type len = Length();
	#pragma omp parallel for SCHEDULE_TINY_OPS
	for (size_type i = 0; i < len; i++)
	{
    real val = absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i));
    absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i)) = 1.0 / val;
	}
	return *this;
}

size_type afb::SubRowVector::Rows( void ) const
{
	return 1;
}

size_type afb::SubRowVector::Columns( void ) const
{
	return absref<afb::VectorView>(mask_).Length();
}

real afb::SubRowVector::operator()( size_type pos ) const
{
  return absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos));
}

real& afb::SubRowVector::operator()( size_type pos )
{
  return absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos));
}

afm::RelativePointer afb::SubRowVector::Create( afm::RelativePointer& targetVector, 
                                                const afm::RelativePointer& mask )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(self));
  new (*ptr) self(targetVector, mask);
  return ptr;
}

void * afb::SubRowVector::operator new( size_t, void *ptr )
{
  return ptr;
}

void afb::SubRowVector::operator delete( void *, void * )
{
  // nothing to do here
}

#include "SubColumnVector.hpp"
#include <math.h>
#include "foundation/memory/pointer.hpp"
#include "foundation/blas/VectorView.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/DimensionMismatchException.hpp"

#define SCHEDULE_TINY_OPS		schedule(dynamic,512)

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

afb::SubColumnVector::SubColumnVector( afm::RelativePointer& targetVector, const afm::RelativePointer& mask ) :
  targetVector_(targetVector), mask_(mask)
{
  // nothing to do here
}

afb::SubColumnVector::~SubColumnVector( void )
{
  // nothing to do here
}

real afb::SubColumnVector::GetElement( size_type pos ) const
{
  return absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos));
}

afb::SubColumnVector::self& afb::SubColumnVector::SetElement( size_type pos, real value )
{
  absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos)) = value;
  return *this;
}

afb::SubColumnVector::self& afb::SubColumnVector::Accumulate( size_type pos, real value )
{
  absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos)) += value;
  return *this;
}

afb::SubColumnVector::self& afb::SubColumnVector::CopyFrom( const self& vector )
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

afb::SubColumnVector::self& afb::SubColumnVector::CopyFrom( const inner_type& vector )
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

afb::SubColumnVector::self& afb::SubColumnVector::ClearAll( void )
{
  SetAll(0);
  return *this;
}

afb::SubColumnVector::self& afb::SubColumnVector::Scale( real value )
{
  size_type len = Length();

  #pragma omp parallel for SCHEDULE_TINY_OPS
  for (size_type i = 0; i < len; i++)
  {
    absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i)) *= value;
  }
  return *this;
}

afb::SubColumnVector::self& afb::SubColumnVector::SetAll( real value )
{
  size_type len = Length();

  #pragma omp parallel for SCHEDULE_TINY_OPS
  for (size_type i = 0; i < len; i++)
  {
    absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(i)) = value;
  }
  return *this;
}

size_type afb::SubColumnVector::Length( void ) const
{
  return absref<afb::VectorView>(mask_).Length();
}

real afb::SubColumnVector::SelfScalarProduct( void ) const
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

real afb::SubColumnVector::Norm( void ) const
{
  return sqrt(SelfScalarProduct());
}

afb::SubColumnVector::self& afb::SubColumnVector::Invert( void )
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

size_type afb::SubColumnVector::Rows( void ) const
{
  return absref<afb::VectorView>(mask_).Length();
}

size_type afb::SubColumnVector::Columns( void ) const
{
  return 1;
}

real afb::SubColumnVector::operator()( size_type pos ) const
{
  return absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos));
}

real& afb::SubColumnVector::operator()( size_type pos )
{
  return absref<inner_type>(targetVector_)(absref<afb::VectorView>(mask_)(pos));
}

afm::RelativePointer afb::SubColumnVector::Create( afm::RelativePointer& targetVector, 
                                               const afm::RelativePointer& mask )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(self));
  new (*ptr) self(targetVector, mask);
  return ptr;
}

void * afb::SubColumnVector::operator new( size_t, void *ptr )
{
  return ptr;
}

void afb::SubColumnVector::operator delete( void *, void * )
{
  // nothing to do here
}

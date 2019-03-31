#include "RowVector.hpp"
#include <math.h>
#include <assert.h>
#include "blas_omp.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/DimensionMismatchException.hpp"
#include "foundation/memory/FixedStackArena.hpp"

#define ALLOCATE_FROM_STACK   0
#define ALLOCATE_FROM_MODEL   1
#define ALLOCATE_FROM_GLOBAL  2

#if defined (AXIS_NO_MEMORY_ARENA) 
#define GET_DATA()     _data
#else
#define GET_DATA()     sourceMemory_ == ALLOCATE_FROM_STACK? _data : ((real *)*_ptr)
#endif

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

struct afb::RowVector::MemorySource
{
public:
  int SourceId;
};

afb::RowVector::RowVector( const self& other )
{
  Init(other._length, ALLOCATE_FROM_STACK);
#if defined(AXIS_NO_MEMORY_ARENA)
  CopyFromVector(other._data, _length);
#else
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, _length);
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, _length);
    break;
  default:
    assert(!"Invalid memory source in argument of RowVector::ctor()!");
    break;
  }
#endif
}

afb::RowVector::RowVector( size_type length )
{
  Init(length, ALLOCATE_FROM_STACK);
}

afb::RowVector::RowVector( size_type length, real inititalValue )
{
  Init(length, ALLOCATE_FROM_STACK);
  SetAll(inititalValue);
}

afb::RowVector::RowVector( size_type length, const real * const values )
{
  Init(length, ALLOCATE_FROM_STACK);
  CopyFromVector(values, length);
}

afb::RowVector::RowVector( size_type length, const real * const values, size_type elementCount )
{
  Init(length, ALLOCATE_FROM_STACK);
  CopyFromVector(values, elementCount);
}

void afb::RowVector::CopyFromVector( const real * const values, size_type count )
{
  if (!(count > 0 && count <= Length()))
  {
    throw axis::foundation::ArgumentException(_T("Invalid value count for vector initialization."));
  }
  real *data = GET_DATA();
  #pragma omp parallel for shared(count, values) VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < count; ++i)
  {
    data[i] = values[i];
  }

  // initialize remaining positions to zero
  #pragma omp parallel for VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = count; i < _length; ++i)
  {
    data[i] = 0;
  }
}

void afb::RowVector::Init( size_type length, int memorySource )
{
  _data = NULL;	// avoids destructor trying to free an invalid memory location, if errors occur here
  if (!(length > 0))
  {
    throw axis::foundation::ArgumentException(_T("Invalid vector dimension."));
  }
  _length = length;
#if defined(AXIS_NO_MEMORY_ARENA)
  _data = new real[_length];
  sourceMemory_ = memorySource;
#else
  uint64 memorySize = sizeof(real)*length;
  switch (memorySource)
  {
  case ALLOCATE_FROM_STACK:
    _data = (real *)System::StackMemory().Allocate((int)memorySize);
    break;
  case ALLOCATE_FROM_MODEL:
    _ptr = System::ModelMemory().Allocate(memorySize);
    _data = NULL;
    break;
  case ALLOCATE_FROM_GLOBAL:
    _ptr = System::GlobalMemory().Allocate(memorySize);
    _data = NULL;
  default:
    break;
  }
  sourceMemory_ = memorySource;
#endif
}

afb::RowVector::~RowVector( void )
{
#if defined(AXIS_NO_MEMORY_ARENA)
  if (_data != NULL) delete [] _data;
#else
  switch (sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    System::StackMemory().Deallocate(_data);
    break;
  case ALLOCATE_FROM_MODEL:
    System::ModelMemory().Deallocate(_ptr);
  case ALLOCATE_FROM_GLOBAL:
  default:
    System::GlobalMemory().Deallocate(_ptr);
    break;
  }
#endif
}

void afb::RowVector::Destroy( void ) const
{
#if defined(AXIS_NO_MEMORY_ARENA)
  delete this;
#else
  switch (sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    System::StackMemory().Deallocate(_data);
    break;
  case ALLOCATE_FROM_MODEL:
    System::ModelMemory().Deallocate(_ptr);
    break;
  case ALLOCATE_FROM_GLOBAL:
    System::GlobalMemory().Deallocate(_ptr);
    break;
  default:
    assert(!"Invalid memory source in RowVector::Destroy()!");
    break;
  }
#endif
}

real afb::RowVector::operator()( size_type pos ) const
{
  if (!(pos >= 0 && pos < _length))
  {
    throw axis::foundation::OutOfBoundsException(_T("RowVector range out of bounds."));
  }
  real *data = GET_DATA();
  return data[pos];
}

real& afb::RowVector::operator()( size_type pos )
{
  if (!(pos >= 0 && pos < _length))
  {
    throw axis::foundation::OutOfBoundsException(_T("RowVector range out of bounds."));
  }
  real *data = GET_DATA();
  return data[pos];
}

real afb::RowVector::GetElement( size_type pos ) const
{
  if (!(pos >= 0 && pos < _length))
  {
    throw axis::foundation::OutOfBoundsException(_T("RowVector range out of bounds."));
  }
  real *data = GET_DATA();
  return data[pos];
}

afb::RowVector::self& afb::RowVector::SetElement( size_type pos, real value )
{
  if (!(pos >= 0 && pos < _length))
  {
    throw axis::foundation::OutOfBoundsException(_T("RowVector range out of bounds."));
  }
  real *data = GET_DATA();
  data[pos] = value;
  return *this;
}

afb::RowVector::self& afb::RowVector::Accumulate( size_type pos, real value )
{
  if (!(pos >= 0 && pos < _length))
  {
    throw axis::foundation::OutOfBoundsException(_T("RowVector range out of bounds."));
  }
  real *data = GET_DATA();
  data[pos] += value;
  return *this;
}

afb::RowVector::self& afb::RowVector::ClearAll( void )
{
  return SetAll(0);
}

afb::RowVector::self& afb::RowVector::Scale( real value )
{
  real *data = GET_DATA();
  #pragma omp parallel for VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] *= value;
  }
  return *this;
}

afb::RowVector::self& afb::RowVector::SetAll( real value )
{
  real *data = GET_DATA();
  #pragma omp parallel for VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] = value;
  }
  return *this;
}

size_type afb::RowVector::Length( void ) const
{
  return _length;
}

afb::RowVector::self& afb::RowVector::operator=( const self& other )
{
  if (Length() != other.Length())
  {
    throw axis::foundation::DimensionMismatchException(_T("Incompatible vector dimension."));
  }
  const real *d = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    d = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    d = (real *)*other._ptr;
    break;
  default:
    assert(!"Invalid memory source in argument of RowVector::operator=()!");
  }
  CopyFromVector(d, Length());
  return *this;
}

afb::RowVector::self& afb::RowVector::operator+=( const self& other )
{
  if (Length() != other.Length())
  {
    throw axis::foundation::InvalidOperationException(_T("Incompatible vector dimension."));
  }
  const real *d = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    d = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    d = (real *)*other._ptr;
    break;
  default:
    assert(!"Invalid memory source in argument of RowVector::operator+=()!");
  }
  real *data = GET_DATA();
  #pragma omp parallel for VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] += d[i];
  }
  return *this;
}

afb::RowVector::self& afb::RowVector::operator-=( const self& other )
{
  if (Length() != other.Length())
  {
    throw axis::foundation::InvalidOperationException(_T("Incompatible vector dimension."));
  }
  const real *d = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    d = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    d = (real *)*other._ptr;
    break;
  default:
    assert(!"Invalid memory source in argument of RowVector::operator-=()!");
  }
  real *data = GET_DATA();
  #pragma omp parallel for VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] -= d[i];
  }
  return *this;
}

afb::RowVector::self& afb::RowVector::operator*=( real scalar )
{
  return Scale(scalar);
}

afb::RowVector::self& afb::RowVector::operator/=( real scalar )
{
  real *data = GET_DATA();
  #pragma omp parallel for VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] /= scalar;
  }
  return *this;
}

bool afb::RowVector::operator==( const self& other ) const
{
  if (_length != other._length)
  {
    return false;
  }

  const real *d = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    d = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    d = (real *)*other._ptr;
    break;
  default:
    assert(!"Invalid memory source in argument of RowVector::operator-=()!");
  }
  real *data = GET_DATA();
  bool ok = true;
  #pragma omp parallel for reduction(&:ok) VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    ok &= data[i] == d[i];
  }
  return ok;
}

bool axis::foundation::blas::RowVector::operator!=( const self& other ) const
{
  return !(*this == other);
}

real afb::RowVector::SelfScalarProduct( void ) const
{
  real *data = GET_DATA();
  real sp = 0;
  #pragma omp parallel for reduction(+:sp) VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    sp += data[i]*data[i];
  }
  return sp;
}

real afb::RowVector::Norm( void ) const
{
  return sqrt(SelfScalarProduct());
}

afb::RowVector::self& afb::RowVector::Invert( void )
{
  real *data = GET_DATA();
  #pragma omp parallel for VECTOR_SCHEDULE_TINY_OPS
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] = 1 / data[i];
  }
  return *this;
}

size_type afb::RowVector::Columns( void ) const
{
  return _length;
}

size_type afb::RowVector::Rows( void ) const
{
  return 1;
}

afb::RowVector::self& afb::RowVector::CopyFrom( const self& source )
{
  return operator =(source);
}

#if !defined(AXIS_NO_MEMORY_ARENA)
afb::RowVector::RowVector( const self& other, const MemorySource& memorySource )
{
  Init(other._length, memorySource.SourceId);
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, _length);
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, _length);
    break;
  default:
    assert(!"Invalid memory source in argument of RowVector::ctor()!");
    break;
  }
}

afb::RowVector::RowVector( size_type length, const MemorySource& memorySource )
{
  Init(length, memorySource.SourceId);
}

afb::RowVector::RowVector( size_type length, real initialValue, const MemorySource& memorySource )
{
  Init(length, memorySource.SourceId);
  SetAll(initialValue);
}

afb::RowVector::RowVector( size_type length, const real * const values, const MemorySource& memorySource )
{
  Init(length, memorySource.SourceId);
  CopyFromVector(values, length);
}

afb::RowVector::RowVector( size_type length, const real * const values, size_type elementCount, const MemorySource& memorySource )
{
  Init(length, memorySource.SourceId);
  CopyFromVector(values, elementCount);
}

afm::RelativePointer afb::RowVector::Create( const self& other )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(other, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::Create( size_type length )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(length, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::Create( size_type length, real initialValue )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(length, initialValue, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::Create( size_type length, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(length, values, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::Create( size_type length, const real * const values, size_type elementCount )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(length, values, elementCount, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::CreateFromGlobalMemory( const self& other )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(other, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::CreateFromGlobalMemory( size_type length )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(length, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::CreateFromGlobalMemory( size_type length, real initialValue )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(length, initialValue, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::CreateFromGlobalMemory( size_type length, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(length, values, src);
  return ptr;
}

afm::RelativePointer afb::RowVector::CreateFromGlobalMemory( size_type length, const real * const values, size_type elementCount )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(length, values, elementCount, src);
  return ptr;
}

void * afb::RowVector::operator new( size_t bytes, afm::RelativePointer& ptr, int location )
{
  switch (location)
  {
  case ALLOCATE_FROM_STACK:
    return System::StackMemory().Allocate((int)bytes);
    break;
  case ALLOCATE_FROM_MODEL:
    ptr = System::ModelMemory().Allocate(bytes);
    break;
  case ALLOCATE_FROM_GLOBAL:
    ptr = System::GlobalMemory().Allocate(bytes);
    break;
  default:
    assert(!"Invalid memory source in RowVector::operator new()!");
  }
  return *ptr;
}

void afb::RowVector::operator delete( void *ptr, afm::RelativePointer& relPtr, int location )
{
  switch (location)
  {
  case ALLOCATE_FROM_STACK:
    System::StackMemory().Deallocate(ptr);
    break;
  case ALLOCATE_FROM_MODEL:
    System::ModelMemory().Deallocate(relPtr);
    break;
  case ALLOCATE_FROM_GLOBAL:
    System::GlobalMemory().Deallocate(relPtr);
    break;
  default:
    assert(!"Invalid memory source in RowVector::operator delete()!");
  }
}
#endif

#include "RowVector.hpp"
#include <math.h>

#define ALLOCATE_FROM_STACK   0
#define ALLOCATE_FROM_MODEL   1
#define ALLOCATE_FROM_GLOBAL  2

#define GET_DATA()     sourceMemory_ == ALLOCATE_FROM_STACK? _data : ((real *)*_ptr)

namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayfb::RowVector::RowVector( const self& other )
{
  Init(other._length);
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, _length);
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, _length);
    break;
  }
}

GPU_ONLY ayfb::RowVector::RowVector( size_type length )
{
  Init(length);
}

GPU_ONLY ayfb::RowVector::RowVector( size_type length, real inititalValue )
{
  Init(length);
  SetAll(inititalValue);
}

GPU_ONLY ayfb::RowVector::RowVector( size_type length, const real * const values )
{
  Init(length);
  CopyFromVector(values, length);
}

GPU_ONLY ayfb::RowVector::RowVector( size_type length, const real * const values, size_type elementCount )
{
  Init(length);
  CopyFromVector(values, elementCount);
}

GPU_ONLY void ayfb::RowVector::CopyFromVector( const real * const values, size_type count )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < count; ++i)
  {
    data[i] = values[i];
  }
  // initialize remaining positions to zero
  for (size_type i = count; i < _length; ++i)
  {
    data[i] = 0;
  }
}

GPU_ONLY void ayfb::RowVector::Init( size_type length )
{
  _data = new real[_length];
  _length = length;
  sourceMemory_ = ALLOCATE_FROM_STACK;
}

GPU_ONLY ayfb::RowVector::~RowVector( void )
{
  if (_data != NULL) delete [] _data;
}

GPU_ONLY void ayfb::RowVector::Destroy( void ) const
{
  delete this;
}

GPU_ONLY real ayfb::RowVector::operator()( size_type pos ) const
{
  real *data = GET_DATA();
  return data[pos];
}

GPU_ONLY real& ayfb::RowVector::operator()( size_type pos )
{
  real *data = GET_DATA();
  return data[pos];
}

GPU_ONLY real ayfb::RowVector::GetElement( size_type pos ) const
{
  real *data = GET_DATA();
  return data[pos];
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::SetElement( size_type pos, real value )
{
  real *data = GET_DATA();
  data[pos] = value;
  return *this;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::Accumulate( size_type pos, real value )
{
  real *data = GET_DATA();
  data[pos] += value;
  return *this;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::ClearAll( void )
{
  return SetAll(0);
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::Scale( real value )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] *= value;
  }
  return *this;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::SetAll( real value )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] = value;
  }
  return *this;
}

GPU_ONLY size_type ayfb::RowVector::Length( void ) const
{
  return _length;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::operator=( const self& other )
{
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
  }
  CopyFromVector(d, Length());
  return *this;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::operator+=( const self& other )
{
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
  }
  real *data = GET_DATA();
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] += d[i];
  }
  return *this;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::operator-=( const self& other )
{
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
  }
  real *data = GET_DATA();
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] -= d[i];
  }
  return *this;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::operator*=( real scalar )
{
  return Scale(scalar);
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::operator/=( real scalar )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] /= scalar;
  }
  return *this;
}

GPU_ONLY bool ayfb::RowVector::operator==( const self& other ) const
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
  }
  real *data = GET_DATA();
  bool ok = true;
  for (size_type i = 0; i < _length; ++i)
  {
    ok &= data[i] == d[i];
  }
  return ok;
}

GPU_ONLY bool ayfb::RowVector::operator!=( const self& other ) const
{
  return !(*this == other);
}

GPU_ONLY real ayfb::RowVector::SelfScalarProduct( void ) const
{
  real *data = GET_DATA();
  real sp = 0;
  for (size_type i = 0; i < _length; ++i)
  {
    sp += data[i]*data[i];
  }
  return sp;
}

GPU_ONLY real ayfb::RowVector::Norm( void ) const
{
  return sqrt(SelfScalarProduct());
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::Invert( void )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _length; ++i)
  {
    data[i] = 1 / data[i];
  }
  return *this;
}

GPU_ONLY size_type ayfb::RowVector::Columns( void ) const
{
  return _length;
}

GPU_ONLY size_type ayfb::RowVector::Rows( void ) const
{
  return 1;
}

GPU_ONLY ayfb::RowVector::self& ayfb::RowVector::CopyFrom( const self& source )
{
  return operator =(source);
}

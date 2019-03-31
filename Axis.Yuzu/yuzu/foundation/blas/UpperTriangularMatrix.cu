#include "UpperTriangularMatrix.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

#define ALLOCATE_FROM_STACK     0
#define ALLOCATE_FROM_MODEL     1
#define ALLOCATE_FROM_GLOBAL    2

#ifndef GET_DATA
#define GET_DATA()    (sourceMemory_ == ALLOCATE_FROM_STACK)? _data : ((real *)*_ptr)
#endif

namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY void ayfb::UpperTriangularMatrix::Init( size_type count )
{
  _size = count;
  size_type sz = count * (count+1)/2;
  _dataLength = sz;
  _data = new real[sz];
  sourceMemory_ = ALLOCATE_FROM_STACK;
}

GPU_ONLY void ayfb::UpperTriangularMatrix::CopyFromVector( const real * const values, size_type count )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < count; ++i)
  {
    data[i] = values[i];
  }

  // fill unassigned spaces with zeros
  for (size_type i = count; i < _dataLength; ++i)
  {
    data[i] = 0;
  }
}

GPU_ONLY ayfb::UpperTriangularMatrix::UpperTriangularMatrix( const self& other )
{
  Init(other._size);
  real *data = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    data = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    data = (real *)*other._ptr;
    break;
  }
  CopyFromVector(data, _dataLength);
}

GPU_ONLY ayfb::UpperTriangularMatrix::UpperTriangularMatrix( size_type size )
{
  Init(size);
}

GPU_ONLY ayfb::UpperTriangularMatrix::UpperTriangularMatrix( size_type size, real value )
{
  Init(size);
  SetAll(value);
}

GPU_ONLY ayfb::UpperTriangularMatrix::UpperTriangularMatrix( size_type size, const real * const values )
{
  Init(size);
  CopyFromVector(values, _dataLength);
}

GPU_ONLY ayfb::UpperTriangularMatrix::UpperTriangularMatrix( size_type size, const real * const values, size_type count )
{
  Init(size);
  CopyFromVector(values, count);
}

GPU_ONLY ayfb::UpperTriangularMatrix::~UpperTriangularMatrix( void )
{
  if (_data != NULL)
  {
    delete [] _data;
  }
}

GPU_ONLY void ayfb::UpperTriangularMatrix::Destroy( void ) const
{
  delete this;
}

GPU_ONLY real& ayfb::UpperTriangularMatrix::operator()( size_type row, size_type column )
{
  real *data = GET_DATA();
  return data[column * (column+1) / 2 + row];	// data is stored in transposed form
}

GPU_ONLY real ayfb::UpperTriangularMatrix::operator()( size_type row, size_type column ) const
{
  real *data = GET_DATA();
  return (row <= column)? data[column * (column+1) / 2 + row] : 0;
}

GPU_ONLY real ayfb::UpperTriangularMatrix::GetElement( size_type row, size_type column ) const
{
  real *data = GET_DATA();
  return (row <= column)? data[column * (column+1) / 2 + row] : 0;	// data is stored in transposed form
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::SetElement( size_type row, 
                                                                                     size_type column, 
                                                                                     real value )
{
  real *data = GET_DATA();
  data[column * (column+1) / 2 + row] = value;	// data is stored in transposed form
  return *this;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::Accumulate( size_type row, 
                                                                                     size_type column, 
                                                                                     real value )
{
  real *data = GET_DATA();
  data[column * (column+1) / 2 + row] += value;	// data is stored in transposed form
  return *this;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::ClearAll( void )
{
  return SetAll(0);
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::Scale( real value )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _dataLength; ++i)
  {
    data[i] *= value;
  }
  return *this;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::SetAll( real value )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _dataLength; ++i)
  {
    data[i] = value;
  }
  return *this;
}

GPU_ONLY size_type ayfb::UpperTriangularMatrix::Rows( void ) const
{
  return _size;
}

GPU_ONLY size_type ayfb::UpperTriangularMatrix::Columns( void ) const
{
  return _size;
}

GPU_ONLY size_type ayfb::UpperTriangularMatrix::ElementCount( void ) const
{
  return _size * _size;
}

GPU_ONLY size_type ayfb::UpperTriangularMatrix::TriangularCount( void ) const
{
  return _dataLength;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::operator=( const self& other )
{
  real *data = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    data = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    data = (real *)*other._ptr;
    break;
  }
  CopyFromVector(data, _dataLength);
  return *this;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::operator+=( const self& other )
{
  const real * d = NULL;
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
  for (size_type i = 0; i < _dataLength; ++i)
  {
    data[i] += d[i];
  }
  return *this;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::operator-=( const self& other )
{
  const real * d = NULL;
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
  for (size_type i = 0; i < _dataLength; ++i)
  {
    data[i] -= d[i];
  }
  return *this;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::operator*=( real scalar )
{
  return Scale(scalar);
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::operator/=( real scalar )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _dataLength; ++i)
  {
    data[i] /= scalar;
  }
  return *this;
}

GPU_ONLY bool ayfb::UpperTriangularMatrix::IsSquare( void ) const
{
  return true;
}

GPU_ONLY real ayfb::UpperTriangularMatrix::Trace( void ) const
{
  real t = 0;
  for (size_type i = 0; i < _size; ++i)
  {
    t += operator()(i,i);
  }
  return t;
}

GPU_ONLY ayfb::UpperTriangularMatrix::self& ayfb::UpperTriangularMatrix::CopyFrom( const self& source )
{
  real *data = GET_DATA();
  for (size_type i = 0; i < _size; ++i)
  {
    for (size_type j = i; j < _size; ++j)
    {
      size_type pos = (j+1)*j/2 + i;
      data[pos] = source(i,j);
    }
  }
  return *this;
}

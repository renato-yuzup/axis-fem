#include "LowerTriangularMatrix.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

#define ALLOCATE_FROM_STACK     0
#define ALLOCATE_FROM_MODEL     1
#define ALLOCATE_FROM_GLOBAL    2

#ifndef GET_DATA
#define GET_DATA()    (sourceMemory_ == ALLOCATE_FROM_STACK)? _data : ((real *)*_ptr)
#endif

namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;


GPU_ONLY void ayfb::LowerTriangularMatrix::Init( size_type count )
{
	_size = count;
	size_type sz = count * (count+1)/2;
	_dataLength = sz;
	_data = new real[sz];
  sourceMemory_ = ALLOCATE_FROM_STACK;
}

GPU_ONLY void ayfb::LowerTriangularMatrix::CopyFromVector( const real * const values, size_type count )
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

GPU_ONLY ayfb::LowerTriangularMatrix::LowerTriangularMatrix( const self& other )
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

GPU_ONLY ayfb::LowerTriangularMatrix::LowerTriangularMatrix( size_type size )
{
	Init(size);
}

GPU_ONLY ayfb::LowerTriangularMatrix::LowerTriangularMatrix( size_type size, real value )
{
	Init(size);
	SetAll(value);
}

GPU_ONLY ayfb::LowerTriangularMatrix::LowerTriangularMatrix( size_type size, const real * const values )
{
	Init(size);
	CopyFromVector(values, _dataLength);
}

GPU_ONLY ayfb::LowerTriangularMatrix::LowerTriangularMatrix( size_type size, const real * const values, size_type count )
{
	Init(size);
	CopyFromVector(values, count);
}

GPU_ONLY ayfb::LowerTriangularMatrix::~LowerTriangularMatrix( void )
{
  delete [] _data;
}

GPU_ONLY void ayfb::LowerTriangularMatrix::Destroy( void ) const
{
	delete this;
}

GPU_ONLY real& ayfb::LowerTriangularMatrix::operator()( size_type row, size_type column )
{
  real *data = GET_DATA();
	return data[row * (row+1) / 2 + column];
}

GPU_ONLY real ayfb::LowerTriangularMatrix::operator()( size_type row, size_type column ) const
{
  real *data = GET_DATA();
	return (row >= column)? data[row * (row+1) / 2 + column] : 0;
}

GPU_ONLY real ayfb::LowerTriangularMatrix::GetElement( size_type row, size_type column ) const
{
  real *data = GET_DATA();
	return (row >= column)? data[row * (row+1) / 2 + column] : 0;
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::SetElement( size_type row, size_type column, real value )
{
  real *data = GET_DATA();
	data[row * (row+1) / 2 + column] = value;
	return *this;
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::Accumulate( size_type row, size_type column, real value )
{
  real *data = GET_DATA();
	data[row * (row+1) / 2 + column] += value;
	return *this;
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::ClearAll( void )
{
	return SetAll(0);
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::Scale( real value )
{
  real *data = GET_DATA();
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] *= value;
	}
	return *this;
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::SetAll( real value )
{
  real *data = GET_DATA();
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] = value;
	}
	return *this;
}

GPU_ONLY size_type ayfb::LowerTriangularMatrix::Rows( void ) const
{
	return _size;
}

GPU_ONLY size_type ayfb::LowerTriangularMatrix::Columns( void ) const
{
	return _size;
}

GPU_ONLY size_type ayfb::LowerTriangularMatrix::ElementCount( void ) const
{
	return _size * _size;
}

GPU_ONLY size_type ayfb::LowerTriangularMatrix::TriangularCount( void ) const
{
	return _dataLength;
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::operator=( const self& other )
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

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::operator+=( const self& other )
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

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::operator-=( const self& other )
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

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::operator*=( real scalar )
{
	return static_cast<self&>(Scale(scalar));
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::operator/=( real scalar )
{
  real *data = GET_DATA();
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] /= scalar;
	}
	return *this;
}

GPU_ONLY bool ayfb::LowerTriangularMatrix::IsSquare( void ) const
{
	return true;
}

GPU_ONLY real ayfb::LowerTriangularMatrix::Trace( void ) const
{
	real t = 0;
	for (size_type i = 0; i < _size; ++i)
	{
		t += operator()(i,i);
	}
	return t;
}

GPU_ONLY ayfb::LowerTriangularMatrix::self& ayfb::LowerTriangularMatrix::CopyFrom( const self& source )
{
  real *data = GET_DATA();
	for (size_type i = 0; i < _size; ++i)
	{
		for (size_type j = 0; j <= i; ++j)
		{
			data[i * (i+1) / 2 + j] = source(i,j);
		}
	}
	return *this;
}

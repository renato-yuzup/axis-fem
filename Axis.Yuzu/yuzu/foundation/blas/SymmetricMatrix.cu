#include "SymmetricMatrix.hpp"
#include "yuzu/foundation/memory/HeapBlockArena.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

#define ALLOCATE_FROM_STACK   0
#define ALLOCATE_FROM_MODEL   1
#define ALLOCATE_FROM_GLOBAL  2

#ifndef GET_DATA
#define GET_DATA()  sourceMemory_ == ALLOCATE_FROM_STACK? _data : ((real *)*_ptr)
#endif

namespace ayfm = axis::yuzu::foundation::memory;
namespace ayfb = axis::yuzu::foundation::blas;

GPU_ONLY ayfb::SymmetricMatrix::SymmetricMatrix( size_type size )
{
	Init(size);
}

GPU_ONLY ayfb::SymmetricMatrix::SymmetricMatrix( size_type size, real value )
{
	Init(size);
	SetAll(value);
}

GPU_ONLY ayfb::SymmetricMatrix::SymmetricMatrix( size_type size, const real * const values )
{
	Init(size);
	CopyFromVector(values, _dataLength);
}

GPU_ONLY ayfb::SymmetricMatrix::SymmetricMatrix( size_type size, const real * const values, size_type count )
{
	Init(size);
	CopyFromVector(values, count);
}

GPU_ONLY ayfb::SymmetricMatrix::SymmetricMatrix( const self& other )
{
  size_type n = other._size;
  Init(n);
  const real * d = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    d = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    d = (real *)*other._ptr;
  }
  CopyFromVector(d, _dataLength);
}

GPU_ONLY void ayfb::SymmetricMatrix::Init( size_type count )
{
  size_type n = count;
  _dataLength = n * (n+1) / 2;
  _size = n;
  _data = new real[_dataLength];
  sourceMemory_ = ALLOCATE_FROM_STACK;
}

GPU_ONLY void ayfb::SymmetricMatrix::CopyFromVector( const real * const values, size_type count )
{
	for (size_type i = 0; i < count; ++i)
	{
		_data[i] = values[i];
	}

	// fill unassigned spaces with zeros
	for (size_type i = count; i < _dataLength; ++i)
	{
		_data[i] = 0;
	}
}

GPU_ONLY ayfb::SymmetricMatrix::~SymmetricMatrix( void )
{
  delete _data;
}

GPU_ONLY void ayfb::SymmetricMatrix::Destroy( void ) const
{
	delete this;
}

GPU_ONLY real& ayfb::SymmetricMatrix::operator()( size_type row, size_type column )
{
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	return data[pos];
}

GPU_ONLY real ayfb::SymmetricMatrix::operator()( size_type row, size_type column ) const
{
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	return data[pos];
}

GPU_ONLY real ayfb::SymmetricMatrix::GetElement( size_type row, size_type column ) const
{
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	return data[pos];
}

GPU_ONLY ayfb::SymmetricMatrix& ayfb::SymmetricMatrix::SetElement( size_type row, size_type column, real value )
{
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	data[pos] = value;
	return *this;
}

GPU_ONLY ayfb::SymmetricMatrix& ayfb::SymmetricMatrix::Accumulate( size_type row, size_type column, real value )
{
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	data[pos] += value;
	return *this;
}

GPU_ONLY ayfb::SymmetricMatrix& ayfb::SymmetricMatrix::ClearAll( void )
{
	return SetAll(0);
}

GPU_ONLY ayfb::SymmetricMatrix& ayfb::SymmetricMatrix::Scale( real value )
{
  real *data = GET_DATA();
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] *= value;
	}
	return *this;
}

GPU_ONLY ayfb::SymmetricMatrix& ayfb::SymmetricMatrix::SetAll( real value )
{
  real *data = GET_DATA();
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] = value;
	}
	return *this;
}

GPU_ONLY size_type ayfb::SymmetricMatrix::Rows( void ) const
{
	return _size;
}

GPU_ONLY size_type ayfb::SymmetricMatrix::Columns( void ) const
{
	return _size;
}

GPU_ONLY size_type ayfb::SymmetricMatrix::ElementCount( void ) const
{
	return _size*_size;
}

GPU_ONLY size_type ayfb::SymmetricMatrix::SymmetricCount( void ) const
{
	return _dataLength;
}

GPU_ONLY ayfb::SymmetricMatrix::self& ayfb::SymmetricMatrix::operator=( const self& other )
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
	CopyFromVector(d, _dataLength);
	return *this;
}

GPU_ONLY ayfb::SymmetricMatrix::self& ayfb::SymmetricMatrix::operator+=( const self& other )
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

GPU_ONLY ayfb::SymmetricMatrix::self& ayfb::SymmetricMatrix::operator-=( const self& other )
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

GPU_ONLY ayfb::SymmetricMatrix::self& ayfb::SymmetricMatrix::operator*=( real scalar )
{
	return (self&)Scale(scalar);
}

GPU_ONLY ayfb::SymmetricMatrix::self& ayfb::SymmetricMatrix::operator/=( real scalar )
{
  real *data = GET_DATA();
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] /= scalar;
	}
	return *this;
}

GPU_ONLY bool ayfb::SymmetricMatrix::IsSquare( void ) const
{
	return true;
}

GPU_ONLY real ayfb::SymmetricMatrix::Trace( void ) const
{
	real t = 0;
	for (size_type i = 0; i < _size; ++i)
	{
		t += operator()(i,i);
	}
	return t;
}

GPU_ONLY ayfb::SymmetricMatrix& ayfb::SymmetricMatrix::CopyFrom( const self& source )
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

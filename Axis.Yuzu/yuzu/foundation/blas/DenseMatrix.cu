#include "DenseMatrix.hpp"

#define LINEARIZE(row,col)		row * _columns + col

#define ALLOCATE_FROM_STACK   0
#define ALLOCATE_FROM_MODEL   1
#define ALLOCATE_FROM_GLOBAL  2

#define MATRIX_DATA()     sourceMemory_ == ALLOCATE_FROM_STACK? _data : ((real *)*_ptr)

namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayfb::DenseMatrix::DenseMatrix( const self& other )
{
	Init(other._rows, other._columns);
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, _rows*_columns);
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, _rows*_columns);
    break;
  }
}

GPU_ONLY ayfb::DenseMatrix::DenseMatrix( size_type rows, size_type columns )
{
	Init(rows, columns);
}

GPU_ONLY ayfb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, real initialValue )
{
	Init(rows, columns);
	SetAll(initialValue);
}

GPU_ONLY ayfb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, const real * const values )
{
	Init(rows, columns);
	CopyFromVector(values, rows*columns);
}

GPU_ONLY ayfb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, const real * const values, size_type elementCount )
{
	Init(rows, columns);
	CopyFromVector(values, elementCount);
}

GPU_ONLY void ayfb::DenseMatrix::CopyFromVector( const real * const values, size_type count )
{
  real *data = MATRIX_DATA();
	for (size_type i = 0; i < count; ++i)
	{
		data[i] = values[i];
	}
	
	// initialize remaining positions to zero
	size_type pos_count = _rows*_columns;
	for (size_type i = count; i < pos_count; ++i)
	{
		data[i] = 0;
	}
}

GPU_ONLY void ayfb::DenseMatrix::Init( size_type rows, size_type columns )
{
  _data = new real[rows*columns];
  _rows = rows;  _columns = columns;
  sourceMemory_ = ALLOCATE_FROM_STACK;
}

GPU_ONLY ayfb::DenseMatrix::~DenseMatrix( void )
{
  if (_data != NULL)
  {
    delete [] _data;
  }
}

GPU_ONLY real ayfb::DenseMatrix::operator()( size_type row, size_type column ) const
{
  real *data = MATRIX_DATA();
	return data[LINEARIZE(row, column)];
}

GPU_ONLY real& ayfb::DenseMatrix::operator()( size_type row, size_type column )
{
  real *data = MATRIX_DATA();
	return data[LINEARIZE(row, column)];
}

GPU_ONLY real ayfb::DenseMatrix::GetElement( size_type row, size_type column ) const
{
  real *data = MATRIX_DATA();
	return data[LINEARIZE(row, column)];
}

GPU_ONLY ayfb::DenseMatrix& ayfb::DenseMatrix::SetElement( size_type row, size_type column, real value )
{
  real *data = MATRIX_DATA();
	data[LINEARIZE(row, column)] = value;
	return *this;
}

GPU_ONLY ayfb::DenseMatrix& ayfb::DenseMatrix::Accumulate( size_type row, size_type column, real value )
{
  real *data = MATRIX_DATA();
	data[LINEARIZE(row, column)] += value;
	return *this;
}

GPU_ONLY ayfb::DenseMatrix& ayfb::DenseMatrix::ClearAll( void )
{
	SetAll(0);
	return *this;
}

GPU_ONLY ayfb::DenseMatrix& ayfb::DenseMatrix::Scale( real value )
{
	size_type sz = ElementCount();
  real *data = MATRIX_DATA();
	for (size_type i = 0; i < sz; ++i)
	{
		data[i] *= value;
	}
	return *this;
}

GPU_ONLY ayfb::DenseMatrix& ayfb::DenseMatrix::SetAll( real value )
{
	size_type sz = ElementCount();
  real *data = MATRIX_DATA();
	for (size_type i = 0; i < sz; ++i)
	{
		data[i] = value;
	}
	return *this;
}

GPU_ONLY size_type ayfb::DenseMatrix::Rows( void ) const
{
	return _rows;
}

GPU_ONLY size_type ayfb::DenseMatrix::Columns( void ) const
{
	return _columns;
}

GPU_ONLY size_type ayfb::DenseMatrix::ElementCount( void ) const
{
	return Rows() * Columns();
}

GPU_ONLY bool ayfb::DenseMatrix::IsSquare( void ) const
{
	return _rows == _columns;
}

GPU_ONLY ayfb::DenseMatrix::self& ayfb::DenseMatrix::operator=( const self& other )
{
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, ElementCount());
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, ElementCount());
    break;
  }
	return *this;
}

GPU_ONLY ayfb::DenseMatrix::self& ayfb::DenseMatrix::operator+=( const self& other )
{
  real *data = MATRIX_DATA();
	for (size_type i = 0; i < _rows; ++i)
	{
    for (size_type j = 0; j < _columns; ++j)
    {
      data[LINEARIZE(i,j)] += other(i,j);
    }
	}
	return *this;
}

GPU_ONLY ayfb::DenseMatrix::self& ayfb::DenseMatrix::operator-=( const self& other )
{
  real *data = MATRIX_DATA();
	for (size_type i = 0; i < _rows; ++i)
	{
    for (size_type j = 0; j < _columns; ++j)
    {
      data[LINEARIZE(i,j)] -= other(i,j);
    }
	}
	return *this;
}

GPU_ONLY ayfb::DenseMatrix::self& ayfb::DenseMatrix::operator*=( real scalar )
{
	return Scale(scalar);
}

GPU_ONLY ayfb::DenseMatrix::self& ayfb::DenseMatrix::operator/=( real scalar )
{
	// doing this way, instead of calling Scale() method,
	// reduces round-off errors
  real *data = MATRIX_DATA();
	size_type sz = ElementCount();
	for (size_type i = 0; i < sz; ++i)
	{
		data[i] /= scalar;
	}
	return *this;
}

GPU_ONLY void ayfb::DenseMatrix::Destroy( void ) const
{
	delete this;
}

GPU_ONLY real ayfb::DenseMatrix::Trace( void ) const
{
	real t = 0;
  real *data = MATRIX_DATA();
	for (size_type i = 0; i < _rows; ++i)
	{
		t += data[LINEARIZE(i,i)];
	}
	return t;
}

GPU_ONLY ayfb::DenseMatrix& ayfb::DenseMatrix::CopyFrom( const self& source )
{
  real *data = MATRIX_DATA();
	for (size_type i = 0; i < _rows; ++i)
	{
		for (size_type j = 0; j < _rows; ++j)
		{
			data[LINEARIZE(i,j)] = source(i,j);
		}
	}
	return *this;
}

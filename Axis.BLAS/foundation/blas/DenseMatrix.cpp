#include "DenseMatrix.hpp"
#include <assert.h>
#include "blas_omp.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/FixedStackArena.hpp"
#include "foundation/memory/HeapStackArena.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/DimensionMismatchException.hpp"
#include "foundation/NotSupportedException.hpp"

#define LINEARIZE(row,col)		row * _columns + col

#define ALLOCATE_FROM_STACK   0
#define ALLOCATE_FROM_MODEL   1
#define ALLOCATE_FROM_GLOBAL  2

#if defined (AXIS_NO_MEMORY_ARENA) 
#define MATRIX_DATA()     _data
#else
#define MATRIX_DATA()     sourceMemory_ == ALLOCATE_FROM_STACK? _data : ((real *)*_ptr)
#endif

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

struct afb::DenseMatrix::MemorySource
{
public:
  int SourceId;
};

afb::DenseMatrix::DenseMatrix( const self& other )
{
	Init(other._rows, other._columns, ALLOCATE_FROM_STACK);
#if defined(AXIS_NO_MEMORY_ARENA)
	CopyFromVector(other._data, _rows*_columns);
#else
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, _rows*_columns);
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, _rows*_columns);
    break;
  default:
    assert(!"Invalid memory source in argument of DenseMatrix::ctor()!");
    break;
  }
#endif
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns )
{
	Init(rows, columns, ALLOCATE_FROM_STACK);
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, real initialValue )
{
	Init(rows, columns, ALLOCATE_FROM_STACK);
	SetAll(initialValue);
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, const real * const values )
{
	Init(rows, columns, ALLOCATE_FROM_STACK);
	CopyFromVector(values, rows*columns);
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, const real * const values, size_type elementCount )
{
	Init(rows, columns, ALLOCATE_FROM_STACK);
	CopyFromVector(values, elementCount);
}

#if !defined(AXIS_NO_MEMORY_ARENA)
afb::DenseMatrix::DenseMatrix( const self& other, const MemorySource& alllocationSource )
{
  Init(other._rows, other._columns, alllocationSource.SourceId);
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, _rows*_columns);
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, _rows*_columns);
    break;
  default:
    assert(!"Invalid memory source in argument of DenseMatrix::ctor()!");
    break;
  }
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, const MemorySource& alllocationSource )
{
  Init(rows, columns, alllocationSource.SourceId);
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, real initialValue, const MemorySource& alllocationSource )
{
  Init(rows, columns, alllocationSource.SourceId);
  SetAll(initialValue);
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, const real * const values, const MemorySource& alllocationSource )
{
  Init(rows, columns, alllocationSource.SourceId);
  CopyFromVector(values, rows*columns);
}

afb::DenseMatrix::DenseMatrix( size_type rows, size_type columns, const real * const values, size_type elementCount, const MemorySource& alllocationSource )
{
  Init(rows, columns, alllocationSource.SourceId);
  CopyFromVector(values, elementCount);
}
#endif

void afb::DenseMatrix::CopyFromVector( const real * const values, size_type count )
{
	if (!(count > 0 && count <= ElementCount()))
	{
		throw axis::foundation::ArgumentException(_T("Invalid value count for matrix initialization."));
	}
  real *data = MATRIX_DATA();

	#pragma omp parallel for shared(count, values) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < count; ++i)
	{
		data[i] = values[i];
	}
	
	// initialize remaining positions to zero
	size_type pos_count = _rows*_columns;
	#pragma omp parallel for shared(pos_count) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = count; i < pos_count; ++i)
	{
		data[i] = 0;
	}
}

void afb::DenseMatrix::Init( size_type rows, size_type columns, int alllocationSource )
{
	_data = NULL;	// avoids destructor trying to free an invalid memory location, if errors occur here
	if (!(rows > 0 && columns > 0))
	{
		throw axis::foundation::ArgumentException(_T("Invalid matrix dimension."));
	}
	_rows = rows;  _columns = columns;
#if defined AXIS_NO_MEMORY_ARENA
	_data = new real[_rows*_columns];
  sourceMemory_ = alllocationSource;
#else
  switch (alllocationSource)
  {
  case ALLOCATE_FROM_STACK:
    _data = (real *)System::StackMemory().Allocate(sizeof(real)*rows*columns);
    break;
  case ALLOCATE_FROM_MODEL:
    _ptr = System::ModelMemory().Allocate(sizeof(real)*rows*columns);
    _data = NULL;
    break;
  case ALLOCATE_FROM_GLOBAL: 
    _ptr = System::GlobalMemory().Allocate(sizeof(real)*rows*columns);
    _data = NULL;
    break;
  default:
    assert(!"Undefined allocation source in DenseMatrix::Init()!");
  }
  sourceMemory_ = alllocationSource;
#endif
}

afb::DenseMatrix::~DenseMatrix( void )
{
#if defined(AXIS_NO_MEMORY_ARENA)
  if (_data != NULL)
  {
    delete [] _data;
  }
#else
  switch (sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    if (_data != NULL)
    {
      System::StackMemory().Deallocate(_data);
    }
    break;
  case ALLOCATE_FROM_MODEL:
    System::ModelMemory().Deallocate(_ptr);
    break;
  case ALLOCATE_FROM_GLOBAL: 
    System::GlobalMemory().Deallocate(_ptr);
    break;
  default:
    assert(!"DenseMatrix memory is corrupted!");
  }
#endif
}

real afb::DenseMatrix::operator()( size_type row, size_type column ) const
{
	if (!(row >=0 && column >=0 && row < _rows && column < _columns))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds."));
	}
  real *data = MATRIX_DATA();
	return data[LINEARIZE(row, column)];
}

real& afb::DenseMatrix::operator()( size_type row, size_type column )
{
	if (!(row >=0 && column >=0 && row < _rows && column < _columns))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds."));
	}
  real *data = MATRIX_DATA();
	return data[LINEARIZE(row, column)];
}

real afb::DenseMatrix::GetElement( size_type row, size_type column ) const
{
	if (!(row >=0 && column >=0 && row < _rows && column < _columns))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds."));
	}
  real *data = MATRIX_DATA();
	return data[LINEARIZE(row, column)];
}

afb::DenseMatrix& afb::DenseMatrix::SetElement( size_type row, size_type column, real value )
{
	if (!(row >=0 && column >=0 && row < _rows && column < _columns))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds."));
	}
  real *data = MATRIX_DATA();
	data[LINEARIZE(row, column)] = value;
	return *this;
}

afb::DenseMatrix& afb::DenseMatrix::Accumulate( size_type row, size_type column, real value )
{
	if (!(row >=0 && column >=0 && row < _rows && column < _columns))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds."));
	}
  real *data = MATRIX_DATA();
	data[LINEARIZE(row, column)] += value;
	return *this;
}

afb::DenseMatrix& afb::DenseMatrix::ClearAll( void )
{
	SetAll(0);
	return *this;
}

afb::DenseMatrix& afb::DenseMatrix::Scale( real value )
{
	size_type sz = ElementCount();
  real *data = MATRIX_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < sz; ++i)
	{
		data[i] *= value;
	}
	return *this;
}

afb::DenseMatrix& afb::DenseMatrix::SetAll( real value )
{
	size_type sz = ElementCount();
  real *data = MATRIX_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < sz; ++i)
	{
		data[i] = value;
	}
	return *this;
}

size_type afb::DenseMatrix::Rows( void ) const
{
	return _rows;
}

size_type afb::DenseMatrix::Columns( void ) const
{
	return _columns;
}

size_type afb::DenseMatrix::ElementCount( void ) const
{
	return Rows() * Columns();
}

bool afb::DenseMatrix::IsSquare( void ) const
{
	return _rows == _columns;
}

afb::DenseMatrix::self& afb::DenseMatrix::operator=( const self& other )
{
	if (other.Rows() != _rows || other.Columns() != _columns)
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
  CopyFromVector(other._data, ElementCount());
#else
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    CopyFromVector(other._data, ElementCount());
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    CopyFromVector((real *)*other._ptr, ElementCount());
    break;
  default:
    assert(!"Invalid memory source in argument of DenseMatrix::ctor()!");
    break;
  }
#endif
	return *this;
}

afb::DenseMatrix::self& afb::DenseMatrix::operator+=( const self& other )
{
	if (other.Rows() != _rows || other.Columns() != _columns)
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
  real *data = MATRIX_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _rows; ++i)
	{
	  #pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
    for (size_type j = 0; j < _columns; ++j)
    {
      data[LINEARIZE(i,j)] += other(i,j);
    }
	}
	return *this;
}

afb::DenseMatrix::self& afb::DenseMatrix::operator-=( const self& other )
{
	if (other.Rows() != _rows || other.Columns() != _columns)
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
  real *data = MATRIX_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _rows; ++i)
	{
	  #pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
    for (size_type j = 0; j < _columns; ++j)
    {
      data[LINEARIZE(i,j)] -= other(i,j);
    }
	}
	return *this;
}

afb::DenseMatrix::self& afb::DenseMatrix::operator*=( real scalar )
{
	return Scale(scalar);
}

afb::DenseMatrix::self& afb::DenseMatrix::operator/=( real scalar )
{
	if (scalar == 0)
	{
		throw axis::foundation::ArgumentException(_T("Division by zero."));
	}
	// doing this way, instead of calling Scale() method,
	// reduces round-off errors
  real *data = MATRIX_DATA();
	size_type sz = ElementCount();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < sz; ++i)
	{
		data[i] /= scalar;
	}
	return *this;
}

void afb::DenseMatrix::Destroy( void ) const
{
#if defined(AXIS_NO_MEMORY_ARENA)
	delete this;
#else
  switch (sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    if (_data != NULL)
    {
      System::StackMemory().Deallocate(_data);
    }
    break;
  case ALLOCATE_FROM_MODEL:
    System::ModelMemory().Deallocate(_ptr);
    break;
  case ALLOCATE_FROM_GLOBAL: 
    System::GlobalMemory().Deallocate(_ptr);
    break;
  default:
    assert(!"DenseMatrix memory is corrupted!");
  }
#endif
}

real afb::DenseMatrix::Trace( void ) const
{
	real t = 0;

	// fail if matrix is not square
	if (!IsSquare())
	{
		throw axis::foundation::NotSupportedException(_T("Operation available only to square matrices."));
	}
  real *data = MATRIX_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _rows; ++i)
	{
		t += data[LINEARIZE(i,i)];
	}
	return t;
}

afb::DenseMatrix& afb::DenseMatrix::CopyFrom( const self& source )
{
	if (source.Rows() != Rows() || source.Columns() != Columns())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
  real *data = MATRIX_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _rows; ++i)
	{
		#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
		for (size_type j = 0; j < _rows; ++j)
		{
			data[LINEARIZE(i,j)] = source(i,j);
		}
	}
	return *this;
}

afm::RelativePointer afb::DenseMatrix::Create( const self& other )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) afb::DenseMatrix(other, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::Create( size_type rows, size_type columns )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) afb::DenseMatrix(rows, columns, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::Create( size_type rows, size_type columns, real initialValue )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) afb::DenseMatrix(rows, columns, initialValue, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::Create( size_type rows, size_type columns, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) afb::DenseMatrix(rows, columns, values, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::Create( size_type rows, size_type columns, const real * const values, size_type elementCount )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) afb::DenseMatrix(rows, columns, values, elementCount, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::CreateFromGlobalMemory( const self& other )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_GLOBAL) afb::DenseMatrix(other, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::CreateFromGlobalMemory( size_type rows, size_type columns )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) afb::DenseMatrix(rows, columns, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::CreateFromGlobalMemory( size_type rows, size_type columns, real initialValue )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) afb::DenseMatrix(rows, columns, initialValue, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::CreateFromGlobalMemory( size_type rows, size_type columns, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) afb::DenseMatrix(rows, columns, values, src);
  return ptr;
}

afm::RelativePointer afb::DenseMatrix::CreateFromGlobalMemory( size_type rows, size_type columns, const real * const values, size_type elementCount )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) afb::DenseMatrix(rows, columns, values, elementCount, src);
  return ptr;
}

void * afb::DenseMatrix::operator new( size_t bytes, afm::RelativePointer& ptr, int location )
{
  switch (location)
  {
  case ALLOCATE_FROM_MODEL: // allocate from model memory
    ptr = System::ModelMemory().Allocate(bytes);
    break;
  case ALLOCATE_FROM_GLOBAL: // allocate from global memory
    ptr = System::GlobalMemory().Allocate(bytes);
    break;
  default:
    assert(!"Invalid parameter passed to DenseMatrix::new operator!");
  }
  return *ptr;
}

void afb::DenseMatrix::operator delete( void *ptr, afm::RelativePointer& relPtr, int location )
{
  switch (location)
  {
  case ALLOCATE_FROM_STACK:
    System::StackMemory().Deallocate(ptr);
    break;
  case ALLOCATE_FROM_MODEL: // allocate from model memory
    System::ModelMemory().Deallocate(relPtr);
    break;
  case ALLOCATE_FROM_GLOBAL: // allocate from global memory
    System::GlobalMemory().Deallocate(relPtr);
    break;
  default:
    assert(!"Invalid parameter passed to DenseMatrix::new operator!");
  }
}

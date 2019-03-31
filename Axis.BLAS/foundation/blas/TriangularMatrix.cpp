#include "TriangularMatrix.hpp"
#include <assert.h>
#include "blas_omp.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/FixedStackArena.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/DimensionMismatchException.hpp"
#include "foundation/NotSupportedException.hpp"

#define ALLOCATE_FROM_STACK     0
#define ALLOCATE_FROM_MODEL     1
#define ALLOCATE_FROM_GLOBAL    2

#if defined(AXIS_NO_MEMORY_ARENA) 
#define GET_DATA()    _data
#else
#define GET_DATA()    (sourceMemory_ == ALLOCATE_FROM_STACK)? _data : ((real *)*_ptr)
#endif

namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

template<int T>	real afb::TriangularMatrix<T>::zero = 0;

template<int T>
struct afb::TriangularMatrix<T>::MemorySource
{
public:
  int SourceId;
};

// explicit instantiate template classes
template class afb::TriangularMatrix<0>;
template class afb::TriangularMatrix<1>;


template<int T>
void afb::TriangularMatrix<T>::Init( size_type count, char memorySource )
{
	_data = NULL;	// avoids destructor trying to free an invalid memory location, if errors occur here
	if (!(count > 0))
	{
		throw axis::foundation::ArgumentException(_T("Invalid matrix dimension."));
	}
	_size = count;
	size_type sz = count * (count+1)/2;
	_dataLength = sz;
#if defined(AXIS_NO_MEMORY_ARENA)
	_data = new real[sz];
  sourceMemory_ = memorySource;
#else
  uint64 memorySize = _dataLength*sizeof(real);
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
    break;
  default:
    assert(!"Invalid memory source in TriangularMatrix::Init()!");
  }
  sourceMemory_ = memorySource;
#endif
}

template<int T>
void afb::TriangularMatrix<T>::CopyFromVector( const real * const values, size_type count )
{
	if (!(count > 0 && count <= _dataLength))
	{
		throw axis::foundation::ArgumentException(_T("Invalid value count for matrix initialization."));
	}
  real *data = GET_DATA();
	#pragma omp parallel for shared(values) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < count; ++i)
	{
		data[i] = values[i];
	}

	// fill unassigned spaces with zeros
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = count; i < _dataLength; ++i)
	{
		data[i] = 0;
	}
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( const self& other )
{
	Init(other._size, ALLOCATE_FROM_STACK);
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
  default:
    assert(!"Invalid memory source in TriangularMatrix::ctor()!");
  }
	CopyFromVector(data, _dataLength);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size )
{
	Init(size, ALLOCATE_FROM_STACK);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size, real value )
{
	Init(size, ALLOCATE_FROM_STACK);
	SetAll(value);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size, const real * const values )
{
	Init(size, ALLOCATE_FROM_STACK);
	CopyFromVector(values, _dataLength);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size, const real * const values, size_type count )
{
	Init(size, ALLOCATE_FROM_STACK);
	CopyFromVector(values, count);
}

#if !defined(AXIS_NO_MEMORY_ARENA)
template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( const self& other, const MemorySource& memorySource )
{
  Init(other._size, memorySource.SourceId);
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
  default:
    assert(!"Invalid memory source in TriangularMatrix::ctor()!");
  }
  CopyFromVector(data, _dataLength);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size, real value, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
  SetAll(value);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size, const real * const values, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
  CopyFromVector(values, _dataLength);
}

template<int T>
afb::TriangularMatrix<T>::TriangularMatrix( size_type size, const real * const values, size_type count, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
  CopyFromVector(values, count);
}
#endif

template<int T>
afb::TriangularMatrix<T>::~TriangularMatrix( void )
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
    System::StackMemory().Deallocate(_data);
    break;
  case ALLOCATE_FROM_MODEL:
    System::ModelMemory().Deallocate(_ptr);
    break;
  case ALLOCATE_FROM_GLOBAL:
    System::GlobalMemory().Deallocate(_ptr);
    break;
  default:
    assert(!"Invalid memory source in TriangularMatrix::~TriangularMatrix()!");
  }
#endif
}

template<int T>
void afb::TriangularMatrix<T>::Destroy( void ) const
{
#if defined (AXIS_NO_MEMORY_ARENA)
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
    assert(!"Invalid memory source in TriangularMatrix::~TriangularMatrix()!");
  }
#endif
}

template<>
real& afb::TriangularMatrix<0>::operator()( size_type row, size_type column )	// lower-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size && column <= row))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	return data[row * (row+1) / 2 + column];
}

template<>
real afb::TriangularMatrix<0>::operator()( size_type row, size_type column ) const	// lower-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	return (row >= column)? data[row * (row+1) / 2 + column] : zero;
}
template<>
real& afb::TriangularMatrix<1>::operator()( size_type row, size_type column )	// upper-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size && column >= row))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	return data[column * (column+1) / 2 + row];	// data is stored in transposed form
}

template<>
real afb::TriangularMatrix<1>::operator()( size_type row, size_type column ) const	// upper-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	return (row <= column)? data[column * (column+1) / 2 + row] : zero;	// data is stored in transposed form
}

template<>
real afb::TriangularMatrix<0>::GetElement( size_type row, size_type column ) const			// lower-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	return (row >= column)? data[row * (row+1) / 2 + column] : 0;
}

template<>
typename afb::TriangularMatrix<0>::self& afb::TriangularMatrix<0>::SetElement( size_type row, size_type column, real value )	// lower-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size && column <= row))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	data[row * (row+1) / 2 + column] = value;
	return *this;
}

template<>
typename afb::TriangularMatrix<0>::self& afb::TriangularMatrix<0>::Accumulate( size_type row, size_type column, real value )	// lower-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size && column <= row))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	data[row * (row+1) / 2 + column] += value;
	return *this;
}
template<>
real afb::TriangularMatrix<1>::GetElement( size_type row, size_type column ) const			// upper-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	return (row <= column)? data[column * (column+1) / 2 + row] : 0;	// data is stored in transposed form
}

template<>
typename afb::TriangularMatrix<1>::self& afb::TriangularMatrix<1>::SetElement( size_type row, size_type column, real value )	// upper-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size && column >= row))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	data[column * (column+1) / 2 + row] = value;	// data is stored in transposed form
	return *this;
}

template<>
typename afb::TriangularMatrix<1>::self& afb::TriangularMatrix<1>::Accumulate( size_type row, size_type column, real value )	// upper-triangular matrix specific
{
	if (!(row >= 0 && column >= 0 && row < _size && column < _size && column >= row))
	{
		throw axis::foundation::OutOfBoundsException(_T("Matrix range out of bounds"));
	}
  real *data = GET_DATA();
	data[column * (column+1) / 2 + row] += value;	// data is stored in transposed form
	return *this;
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::ClearAll( void )
{
	return SetAll(0);
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::Scale( real value )
{
  real *data = GET_DATA();
	#pragma omp parallel for shared(value) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] *= value;
	}
	return *this;
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::SetAll( real value )
{
  real *data = GET_DATA();
	#pragma omp parallel for shared(value) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] = value;
	}
	return *this;
}

template<int T>
size_type afb::TriangularMatrix<T>::Rows( void ) const
{
	return _size;
}

template<int T>
size_type afb::TriangularMatrix<T>::Columns( void ) const
{
	return _size;
}

template<int T>
size_type afb::TriangularMatrix<T>::ElementCount( void ) const
{
	return _size * _size;
}

template<int T>
size_type afb::TriangularMatrix<T>::TriangularCount( void ) const
{
	return _dataLength;
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::operator=( const self& other )
{
	if (other._size != _size)
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
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
  default:
    assert(!"Invalid memory source in argument of TriangularMatrix::operator=()!");
  }
	CopyFromVector(data, _dataLength);
	return *this;
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::operator+=( const self& other )
{
	if (other.Rows() != Rows())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
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
  default:
    assert(!"Invalid memory source in argument of TriangularMatrix::operator+=()!");
  }
  real *data = GET_DATA();
	#pragma omp parallel for shared(d) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] += d[i];
	}
	return *this;
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::operator-=( const self& other )
{
	if (other.Rows() != Rows())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
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
  default:
    assert(!"Invalid memory source in argument of TriangularMatrix::operator-=()!");
  }
  real *data = GET_DATA();
	#pragma omp parallel for shared(d) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] -= d[i];
	}
	return *this;
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::operator*=( real scalar )
{
	return static_cast<self&>(Scale(scalar));
}

template<int T>
typename afb::TriangularMatrix<T>::self& afb::TriangularMatrix<T>::operator/=( real scalar )
{
  real *data = GET_DATA();
	#pragma omp parallel for shared(scalar) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] /= scalar;
	}
	return *this;
}

template<int T>
bool afb::TriangularMatrix<T>::IsSquare( void ) const
{
	return true;
}

template<int T>
real afb::TriangularMatrix<T>::Trace( void ) const
{
	real t = 0;
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _size; ++i)
	{
		t += operator()(i,i);
	}
	return t;
}

template<>
typename afb::TriangularMatrix<0>::self& afb::TriangularMatrix<0>::CopyFrom( const self& source )	// lower-triangular
{
	if (source.Rows() != Rows() || source.Columns() != Columns())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
  real *data = GET_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _size; ++i)
	{
		#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
		for (size_type j = 0; j <= i; ++j)
		{
			data[i * (i+1) / 2 + j] = source(i,j);
		}
	}
	return *this;
}

template<>
typename afb::TriangularMatrix<1>::self& afb::TriangularMatrix<1>::CopyFrom( const self& source )	// upper-triangular
{
	if (source.Rows() != Rows() || source.Columns() != Columns())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
  real *data = GET_DATA();
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _size; ++i)
	{
		#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
		for (size_type j = i; j < _size; ++j)
		{
			size_type pos = (j+1)*j/2 + i;
			data[pos] = source(i,j);
		}
	}
	return *this;
}

#if !defined(AXIS_NO_MEMORY_ARENA)
template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::Create( const self& other )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(other, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::Create( size_type size )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(size, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::Create( size_type size, real value )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(size, value, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::Create( size_type size, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(size, values, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::Create( size_type size, const real * const values, size_type count )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) self(size, values, count, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::CreateFromGlobalMemory( const self& other )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(other, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::CreateFromGlobalMemory( size_type size )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(size, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::CreateFromGlobalMemory( size_type size, real value )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(size, value, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::CreateFromGlobalMemory( size_type size, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(size, values, src);
  return ptr;
}

template<int T>
afm::RelativePointer afb::TriangularMatrix<T>::CreateFromGlobalMemory( size_type size, const real * const values, size_type count )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) self(size, values, count, src);
  return ptr;
}

template<int T>
void * afb::TriangularMatrix<T>::operator new( size_t bytes, afm::RelativePointer& ptr, int location )
{
  switch (location)
  {
  case ALLOCATE_FROM_STACK:
    return System::StackMemory().Allocate((int)bytes);
  case ALLOCATE_FROM_MODEL:
    ptr = System::ModelMemory().Allocate(bytes);
    break;
  case ALLOCATE_FROM_GLOBAL:
    ptr = System::GlobalMemory().Allocate(bytes);
    break;
  default:
    assert(!"Invalid memory source in TriangularMatrix::operator new()!");
    break;
  }
  return *ptr;
}

template<int T>
void afb::TriangularMatrix<T>::operator delete( void *ptr, afm::RelativePointer& relPtr, int location )
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
    assert(!"Invalid memory source in TriangularMatrix::operator delete()!");
    break;
  }
}
#endif

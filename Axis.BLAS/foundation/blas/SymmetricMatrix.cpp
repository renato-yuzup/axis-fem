#include "SymmetricMatrix.hpp"
#include "blas_omp.hpp"
#include <assert.h>
#include "foundation/memory/HeapBlockArena.hpp"
#include "System.hpp"
#include "foundation/memory/FixedStackArena.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/DimensionMismatchException.hpp"
#include "foundation/NotSupportedException.hpp"

#define ALLOCATE_FROM_STACK   0
#define ALLOCATE_FROM_MODEL   1
#define ALLOCATE_FROM_GLOBAL  2

#if defined(AXIS_NO_MEMORY_ARENA) 
#define GET_DATA()        _data
#else
#define GET_DATA()        sourceMemory_ == ALLOCATE_FROM_STACK? _data : ((real *)*_ptr)
#endif

namespace afm = axis::foundation::memory;
namespace afb = axis::foundation::blas;

struct afb::SymmetricMatrix::MemorySource
{
public:
  int SourceId;
};

afb::SymmetricMatrix::SymmetricMatrix( const self& other )
{
	size_type n = other._size;
	Init(n, ALLOCATE_FROM_STACK);
#if defined(AXIS_NO_MEMORY_ARENA)
  CopyFromVector(other._data, _dataLength);
#else
  const real * d = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    d = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    d = (real *)*other._ptr;
  default:
    assert(!"Corrupt memory in argument of method SymmetricMatrix::ctor()!");
  }
  CopyFromVector(d, _dataLength);
#endif
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size )
{
	Init(size, ALLOCATE_FROM_STACK);
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size, real value )
{
	Init(size, ALLOCATE_FROM_STACK);
	SetAll(value);
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size, const real * const values )
{
	Init(size, ALLOCATE_FROM_STACK);
	CopyFromVector(values, _dataLength);
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size, const real * const values, size_type count )
{
	Init(size, ALLOCATE_FROM_STACK);
	CopyFromVector(values, count);
}

#if !defined(AXIS_NO_MEMORY_ARENA)
afb::SymmetricMatrix::SymmetricMatrix( const self& other, const MemorySource& memorySource )
{
  size_type n = other._size;
  Init(n, memorySource.SourceId);
  const real * d = NULL;
  switch (other.sourceMemory_)
  {
  case ALLOCATE_FROM_STACK:
    d = other._data;
    break;
  case ALLOCATE_FROM_MODEL:
  case ALLOCATE_FROM_GLOBAL:
    d = (real *)*other._ptr;
  default:
    assert(!"Corrupt memory in argument of method SymmetricMatrix::ctor()!");
  }
  CopyFromVector(d, _dataLength);
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size, real value, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
  SetAll(value);
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size, const real * const values, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
  CopyFromVector(values, _dataLength);
}

afb::SymmetricMatrix::SymmetricMatrix( size_type size, const real * const values, size_type count, const MemorySource& memorySource )
{
  Init(size, memorySource.SourceId);
  CopyFromVector(values, count);
}
#endif

void afb::SymmetricMatrix::Init( size_type count, int memorySource )
{
	_data = NULL;	// avoids destructor trying to free an invalid memory location, if errors occur here
  size_type n = count;
  _dataLength = n * (n+1) / 2;
  _size = n;
  if (!(count > 0))
  {
    throw axis::foundation::ArgumentException(_T("Invalid matrix size."));
  }
#if defined(AXIS_NO_MEMORY_ARENA)
	_data = new real[_dataLength];
  sourceMemory_ = memorySource;
#else
  uint64 memorySize = sizeof(real)*_dataLength;
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
    assert(!"Invalid memory source in SymmetricMatrix::Init()!");
    break;
  }
  sourceMemory_ = memorySource;
#endif
}

void afb::SymmetricMatrix::CopyFromVector( const real * const values, size_type count )
{
	if (!(count > 0 && count <= _dataLength))
	{
		throw axis::foundation::ArgumentException(_T("Invalid value count for matrix initialization."));
	}
	#pragma omp parallel for shared(values) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < count; ++i)
	{
		_data[i] = values[i];
	}

	// fill unassigned spaces with zeros
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = count; i < _dataLength; ++i)
	{
		_data[i] = 0;
	}
}

afb::SymmetricMatrix::~SymmetricMatrix( void )
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
    assert(!"Invalid memory source in SymmetricMatrix::~SymmetricMatrix()!");
    break;
  }
#endif
}

void afb::SymmetricMatrix::Destroy( void ) const
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
    assert(!"Invalid memory source in SymmetricMatrix::Destroy()!");
    break;
  }
#endif
}

real& afb::SymmetricMatrix::operator()( size_type row, size_type column )
{
	if (!(column >= 0 && row >= 0 && column < _size && row < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Invalid matrix range."));
	}
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	return data[pos];
}

real afb::SymmetricMatrix::operator()( size_type row, size_type column ) const
{
	if (!(column >= 0 && row >= 0 && column < _size && row < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Invalid matrix range."));
	}
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	return data[pos];
}

real afb::SymmetricMatrix::GetElement( size_type row, size_type column ) const
{
	if (!(column >= 0 && row >= 0 && column < _size && row < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Invalid matrix range."));
	}
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	return data[pos];
}

afb::SymmetricMatrix& afb::SymmetricMatrix::SetElement( size_type row, size_type column, real value )
{
	if (!(column >= 0 && row >= 0 && column < _size && row < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Invalid matrix range."));
	}
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	data[pos] = value;
	return *this;
}

afb::SymmetricMatrix& afb::SymmetricMatrix::Accumulate( size_type row, size_type column, real value )
{
	if (!(column >= 0 && row >= 0 && column < _size && row < _size))
	{
		throw axis::foundation::OutOfBoundsException(_T("Invalid matrix range."));
	}
	size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
  real *data = GET_DATA();
	data[pos] += value;
	return *this;
}

afb::SymmetricMatrix& afb::SymmetricMatrix::ClearAll( void )
{
	return SetAll(0);
}

afb::SymmetricMatrix& afb::SymmetricMatrix::Scale( real value )
{
  real *data = GET_DATA();
	#pragma omp parallel for shared(value) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] *= value;
	}
	return *this;
}

afb::SymmetricMatrix& afb::SymmetricMatrix::SetAll( real value )
{
  real *data = GET_DATA();
	#pragma omp parallel for shared(value) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] = value;
	}
	return *this;
}

size_type afb::SymmetricMatrix::Rows( void ) const
{
	return _size;
}

size_type afb::SymmetricMatrix::Columns( void ) const
{
	return _size;
}

size_type afb::SymmetricMatrix::ElementCount( void ) const
{
	return _size*_size;
}

size_type afb::SymmetricMatrix::SymmetricCount( void ) const
{
	return _dataLength;
}

afb::SymmetricMatrix::self& afb::SymmetricMatrix::operator=( const self& other )
{
	if (other.Rows() != Rows())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
	const real * const d = other._data;
#else
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
      assert(!"Corrupt memory in argument of method SymmetricMatrix::operator=()!");
  }
#endif
	CopyFromVector(d, _dataLength);
	return *this;
}

afb::SymmetricMatrix::self& afb::SymmetricMatrix::operator+=( const self& other )
{
	if (other.Rows() != Rows())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
  const real * d = other._data;
#else
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
    assert(!"Corrupt memory in argument of method SymmetricMatrix::operator+=()!");
  }
#endif
  real *data = GET_DATA();
	#pragma omp parallel for shared(d) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] += d[i];
	}

	return *this;
}

afb::SymmetricMatrix::self& afb::SymmetricMatrix::operator-=( const self& other )
{
	if (other.Rows() != Rows())
	{
		throw axis::foundation::DimensionMismatchException(_T("Incompatible matrix dimensions."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
  const real * const d = other._data;
#else
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
    assert(!"Corrupt memory in argument of method SymmetricMatrix::operator-=()!");
  }
#endif
  real *data = GET_DATA();
	#pragma omp parallel for shared(d) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] -= d[i];
	}
	return *this;
}

afb::SymmetricMatrix::self& afb::SymmetricMatrix::operator*=( real scalar )
{
	return (self&)Scale(scalar);
}

afb::SymmetricMatrix::self& afb::SymmetricMatrix::operator/=( real scalar )
{
  real *data = GET_DATA();
	#pragma omp parallel for shared(scalar) MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _dataLength; ++i)
	{
		data[i] /= scalar;
	}
	return *this;
}

bool afb::SymmetricMatrix::IsSquare( void ) const
{
	return true;
}

real afb::SymmetricMatrix::Trace( void ) const
{
	real t = 0;
	#pragma omp parallel for MATRIX_SCHEDULE_TINY_OPS
	for (size_type i = 0; i < _size; ++i)
	{
		t += operator()(i,i);
	}
	return t;
}

afb::SymmetricMatrix& afb::SymmetricMatrix::CopyFrom( const self& source )
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
afm::RelativePointer afb::SymmetricMatrix::Create( size_type size )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) SymmetricMatrix(size, src);
  return ptr;
}

afm::RelativePointer afb::SymmetricMatrix::Create( size_type size, real value )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) SymmetricMatrix(size, value, src);
  return ptr;
}

afm::RelativePointer afb::SymmetricMatrix::Create( size_type size, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) SymmetricMatrix(size, values, src);
  return ptr;
}

afm::RelativePointer afb::SymmetricMatrix::Create( size_type size, const real * const values, size_type count )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_MODEL;
  new (ptr, ALLOCATE_FROM_MODEL) SymmetricMatrix(size, values, count, src);
  return ptr;
}

afm::RelativePointer afb::SymmetricMatrix::CreateFromGlobalMemory( size_type size )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) SymmetricMatrix(size, src);
  return ptr;
}

afm::RelativePointer afb::SymmetricMatrix::CreateFromGlobalMemory( size_type size, real value )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) SymmetricMatrix(size, value, src);
  return ptr;
}

afm::RelativePointer afb::SymmetricMatrix::CreateFromGlobalMemory( size_type size, const real * const values )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) SymmetricMatrix(size, values, src);
  return ptr;
}

afm::RelativePointer afb::SymmetricMatrix::CreateFromGlobalMemory( size_type size, const real * const values, size_type count )
{
  afm::RelativePointer ptr;
  MemorySource src; src.SourceId = ALLOCATE_FROM_GLOBAL;
  new (ptr, ALLOCATE_FROM_GLOBAL) SymmetricMatrix(size, values, count, src);
  return ptr;
}

void * afb::SymmetricMatrix::operator new( size_t bytes, afm::RelativePointer& ptr, int location )
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
    assert(!"Memory in SymmetricMatrix is corrupted!");
    break;
  }
  return *ptr;
}

void afb::SymmetricMatrix::operator delete( void *ptr, afm::RelativePointer& relPtr, int location )
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
    assert(!"Memory in SymmetricMatrix is corrupted!");
    break;
  }
}
#endif
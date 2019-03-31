/**********************************************************************************************//**
 * @file	foundation\blas\SymmetricMatrix.hpp
 *
 * @brief	Declares the symmetric matrix class.
 **************************************************************************************************/
#pragma once
#include "foundation/blas/Axis.BLAS.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* @class	SymmetricMatrix
	*
	* @brief	A square matrix which is equal to its transpose.
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @sa	Matrix
	**************************************************************************************************/
class AXISBLAS_API SymmetricMatrix
{
public:

	/**********************************************************************************************//**
		* @brief	Defines an alias for the type of this object.
		**************************************************************************************************/
	typedef SymmetricMatrix self;

	/**********************************************************************************************//**
		* @brief	Copy constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other matrix.
		**************************************************************************************************/
	SymmetricMatrix(const self& other);

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	size	The matrix size.
		**************************************************************************************************/
	SymmetricMatrix(size_type size);

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix and sets all values to a given constant.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	size 	The matrix size.
		* @param	value	The value to assign to each matrix element.
		**************************************************************************************************/
	SymmetricMatrix(size_type size, real value);

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix and initializes all values
		* 			as specified in an initialization array.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	size  	The matrix size.
		* @param	values	The initialization array containing the values in
		* 					a row-major order for the lower symmetric part of
		* 					the matrix.
		**************************************************************************************************/
	SymmetricMatrix(size_type size, const real * const values);

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix and initializes some
		* 			values as specified in an initialization array. Remaining
		* 			values are initialized to zero.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	size  	The matrix size.
		* @param	values	The initialization values in a row-major order
		* 					for the lower symmetric part of the matrix.
		* @param	count 	Number of elements to be initialized by the
		* 					initialization array, starting from index (0,0).
		**************************************************************************************************/
	SymmetricMatrix(size_type size, const real * const values, size_type count);

	/**********************************************************************************************//**
		* @brief	Default destructor for this class.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	~SymmetricMatrix(void);

	/**********************************************************************************************//**
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	void Destroy(void) const;

	/**********************************************************************************************//**
		* @brief	Returns an element of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param   row      The zero-based index of the row where the element is located.
		* @param   column   The zero-based index of the column where the element is located.
		* 				   
		* @return	A read-only reference to the element at the specified coordinates.
		**************************************************************************************************/
	real& operator ()(size_type row, size_type column);

	/**********************************************************************************************//**
		* @brief	Returns an element of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	row		The zero-based index of the row where the element
		* 					is located.
		* @param	column	The zero-based index of the column where the
		* 					element is located.
		* 					
		* @return	A writable reference to the element at the specified
		* 			coordinates.
		**************************************************************************************************/
	real operator ()(size_type row, size_type column) const;

	/**********************************************************************************************//**
		* @brief	Returns an element of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	row   	The zero-based index of the row where the element
		* 					is located.
		* @param	column	The zero-based index of the column where the
		* 					element is located.
		*
		* @return	The element value.
		**************************************************************************************************/
	real GetElement(size_type row, size_type column) const;

	/**********************************************************************************************//**
		* @brief	Assigns a value to an element of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	row   	The zero-based index of the row where the element
		* 					is located.
		* @param	column	The zero-based index of the column where the
		* 					element is located.
		* @param	value 	The value to be assigned to the element.
		*
		* @return	This matrix object.
		**************************************************************************************************/
	self& SetElement(size_type row, size_type column, real value);

	/**********************************************************************************************//**
		* @brief	Increments by a specified amount the value of an element
		* 			of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	row   	The zero-based index of the row where the element
		* 					is located.
		* @param	column	The zero-based index of the column where the
		* 					element is located.
		* @param	value 	The increment value for the element.
		*
		* @return	This matrix object.
		**************************************************************************************************/
	self& Accumulate(size_type row, size_type column, real value);

	/**********************************************************************************************//**
		* @brief	Copies data from the described by source matrix.
		* 			@note	Source matrix must be of the same size in order to
		* 				fulfill this operation. Due to performance issues, no
		* 				symmetry checking is made on source. This operation
		* 				assumes that the matrix is symmetric and copies the
		* 				diagonal and upper part of the source matrix to this
		* 				object only.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	source	Matrix from which to copy all data.
		*
		* @return	This object.
		**************************************************************************************************/
	self& CopyFrom(const self& source);

	/**********************************************************************************************//**
		* @brief	Assigns zeros to all elements of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	This matrix object.
		**************************************************************************************************/
	self& ClearAll(void);

	/**********************************************************************************************//**
		* @brief	Multiplies every element of this matrix by a specified
		* 			factor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	value	The factor by which the elements will be
		* 					multiplied.
		*
		* @return	This matrix object.
		**************************************************************************************************/
	self& Scale(real value);

	/**********************************************************************************************//**
		* @brief	Sets all elements to a specified value.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	value	The value to be assigned to the elements.
		*
		* @return	This matrix object.
		**************************************************************************************************/
	self& SetAll(real value);

	/**********************************************************************************************//**
		* @brief	Returns the number of rows of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-null, positive number representing the number of
		* 			rows.
		**************************************************************************************************/
	size_type Rows(void) const;

	/**********************************************************************************************//**
		* @brief	Returns the number of columns of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-null, positive number representing the number of
		* 			columns.
		**************************************************************************************************/
	size_type Columns(void) const;

	/**********************************************************************************************//**
		* @brief	Returns the total number of addressable elements in this
		* 			matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-null, positive number representing the number of
		* 			addressable elements in this matrix, which actually is
		* 			the product of the number of rows and number of columns.
		**************************************************************************************************/
	size_type ElementCount(void) const;

	/**********************************************************************************************//**
		* @brief	Returns the number of elements in the lower symmetric
		* 			part (including the main diagonal) of the matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	size_type SymmetricCount(void) const;

	/**********************************************************************************************//**
		* @brief	Tells if the column and row count are the same.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	true is both row and column count are the same, false
		* 			otherwise.
		**************************************************************************************************/
	bool IsSquare(void) const;

	/**********************************************************************************************//**
		* @brief	Copies the contents of another matrix to this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other matrix.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	self& operator = (const self& other);

	/**********************************************************************************************//**
		* @brief	Stores in this object the result of A + B, where A is
		* 			this matrix and B is another matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other matrix (B).
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	self& operator += (const self& other);

	/**********************************************************************************************//**
		* @brief	Stores in this object the result of A - B, where A is
		* 			this matrix and B is another matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other matrix (B).
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	self& operator -= (const self& other);

	/**********************************************************************************************//**
		* @brief	Multiplies every matrix element by a given factor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	scalar	The factor.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	self& operator *= (real scalar);

	/**********************************************************************************************//**
		* @brief	Divides every matrix element by a given divisor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	scalar	The divisor.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	self& operator /= (real scalar);

	/**********************************************************************************************//**
		* @brief	Returns the sum of the elements in the main diagonal of
		* 			the matrix.
		* 			@note This operation is valid only for square matrix. It
		* 				might be useful to check for this property using
		* 				IsSquare() method before attempting to execute this
		* 				operation.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A real number which is the trace of this matrix.
		**************************************************************************************************/
	real Trace(void) const;

  static axis::foundation::memory::RelativePointer Create(size_type size);
  static axis::foundation::memory::RelativePointer Create(size_type size, real value);
  static axis::foundation::memory::RelativePointer Create(size_type size, const real * const values);
  static axis::foundation::memory::RelativePointer Create(size_type size, const real * const values, size_type count);
  static axis::foundation::memory::RelativePointer CreateFromGlobalMemory(size_type size);
  static axis::foundation::memory::RelativePointer CreateFromGlobalMemory(size_type size, real value);
  static axis::foundation::memory::RelativePointer CreateFromGlobalMemory(size_type size, const real * const values);
  static axis::foundation::memory::RelativePointer CreateFromGlobalMemory(size_type size, const real * const values, size_type count);
private:
  struct MemorySource;

	real * _data;
  size_type _size;
  size_type _dataLength;
  axis::foundation::memory::RelativePointer _ptr;
  int sourceMemory_;

	SymmetricMatrix(const self& other, const MemorySource& memorySource);
	SymmetricMatrix(size_type size, const MemorySource& memorySource);
	SymmetricMatrix(size_type size, real value, const MemorySource& memorySource);
	SymmetricMatrix(size_type size, const real * const values, const MemorySource& memorySource);
	SymmetricMatrix(size_type size, const real * const values, size_type count, const MemorySource& memorySource);

	/**********************************************************************************************//**
		* @brief	Initializes this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	count	        Number of elements in the symmetric part.
    * @param memorySource  Indicates from where memory should be allocated.
		**************************************************************************************************/
	void Init(size_type count, int memorySource);

	/**********************************************************************************************//**
		* @brief	Initializes this object and copies values from an initialization vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	Pointer to the array of initialization values.
		* @param	count 	Number of elements in the symmetric part.
		**************************************************************************************************/
	void CopyFromVector(const real * const values, size_type count);

  void * operator new (size_t bytes);
  void * operator new (size_t bytes, axis::foundation::memory::RelativePointer& ptr, int location);
  void operator delete (void *ptr, axis::foundation::memory::RelativePointer& relPtr, int location);
};

/**********************************************************************************************//**
	* @typedef	AXISBLAS_API SymmetricMatrix SymmetricSecondTensor
	*
	* @brief	Defines an alias for a symmetric second-order
	* 			tensor, which is represented by a symmetric matrix.
	**************************************************************************************************/
typedef SymmetricMatrix SymmetricSecondTensor;

} } } // namespace axis::foundation::blas


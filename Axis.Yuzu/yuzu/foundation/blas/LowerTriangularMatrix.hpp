/**********************************************************************************************//**
 * @file	foundation/blas/LowerTriangularMatrix.hpp
 *
 * @brief	Declares the triangular matrix class.
 **************************************************************************************************/
#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* @brief	A square matrix in which all elements below or above 
	* 			the main diagonal are always zero.
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @sa	Matrix
	**************************************************************************************************/
class LowerTriangularMatrix
{
public:

	/**********************************************************************************************//**
		* @brief	Defines an alias for the type of this object.
		**************************************************************************************************/
	typedef LowerTriangularMatrix self;

	/**********************************************************************************************//**
		* @brief	Copy constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other matrix.
		**************************************************************************************************/
	GPU_ONLY LowerTriangularMatrix(const self& other);

	/**********************************************************************************************//**
		* @brief	Creates a new triangular matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	size	The matrix size.
		**************************************************************************************************/
	GPU_ONLY LowerTriangularMatrix(size_type size);

	/**********************************************************************************************//**
		* @brief	Creates a new triangular matrix and assigns to every
		* 			element the specified constant.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	size 	The matrix size.
		* @param	value	The value to be assigned to the matrix elements.
		**************************************************************************************************/
	GPU_ONLY LowerTriangularMatrix(size_type size, real value);
	GPU_ONLY LowerTriangularMatrix(size_type size, const real * const values);
	GPU_ONLY LowerTriangularMatrix(size_type size, const real * const values, size_type count);

	/**********************************************************************************************//**
		* @brief	Default destructor for this class.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY ~LowerTriangularMatrix(void);

	/**********************************************************************************************//**
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY void Destroy(void) const;

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
	GPU_ONLY real& operator ()(size_type row, size_type column);

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
	GPU_ONLY real operator ()(size_type row, size_type column) const;

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
	GPU_ONLY real GetElement(size_type row, size_type column) const;

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
	GPU_ONLY self& SetElement(size_type row, size_type column, real value);

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
	GPU_ONLY self& Accumulate(size_type row, size_type column, real value);

	/**********************************************************************************************//**
		* @brief	Copies data from the described by source matrix.
		* 			@note	Source matrix must be of the same size in order to
		* 				fulfill this operation. Due to performance issues, no
		* 				checking is made on source. This operation assumes that
		* 				the matrix is triangular and copies the diagonal and
		* 				upper/lower part of the source matrix to this object
		* 				only.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	source	Matrix from which to copy all data.
		*
		* @return	This object.
		**************************************************************************************************/
	GPU_ONLY self& CopyFrom(const self& source);

	/**********************************************************************************************//**
		* @brief	Assigns zeros to all elements of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	This matrix object.
		**************************************************************************************************/
	GPU_ONLY self& ClearAll(void);

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
	GPU_ONLY self& Scale(real value);

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
	GPU_ONLY self& SetAll(real value);

	/**********************************************************************************************//**
		* @brief	Returns the number of rows of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-null, positive number representing the number of
		* 			rows.
		**************************************************************************************************/
	GPU_ONLY size_type Rows(void) const;

	/**********************************************************************************************//**
		* @brief	Returns the number of columns of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-null, positive number representing the number of
		* 			columns.
		**************************************************************************************************/
	GPU_ONLY size_type Columns(void) const;

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
	GPU_ONLY size_type ElementCount(void) const;

	/**********************************************************************************************//**
		* @brief	Returns the number of writable elements in this matrix,
		* 			that is, the number of elements in lower/upper triangular
		* 			part (including main diagonal) of the matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type TriangularCount(void) const;

	/**********************************************************************************************//**
		* @brief	Tells if the column and row count are the same.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	true is both row and column count are the same, false
		* 			otherwise.
		**************************************************************************************************/
	GPU_ONLY bool IsSquare(void) const;

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
	GPU_ONLY self& operator = (const self& other);

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
	GPU_ONLY self& operator += (const self& other);

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
	GPU_ONLY self& operator -= (const self& other);

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
	GPU_ONLY self& operator *= (real scalar);

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
	GPU_ONLY self& operator /= (real scalar);

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
	GPU_ONLY real Trace(void) const;
private:
	real * _data;
	size_type _dataLength;
	size_type _size;
  axis::yuzu::foundation::memory::RelativePointer _ptr;
  char sourceMemory_;

	/**********************************************************************************************//**
		* @brief	Initializes this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	count	        Number of writable elements in the matrix.
		**************************************************************************************************/
	GPU_ONLY void Init(size_type count);

	/**********************************************************************************************//**
		* @brief	Copies values from a vector to this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	The array of values to be copied.
		* @param	count 	Number of elements to be copied. Assignments will
		* 					be made in a row-major order starting from index
		* 					(0,0).
		**************************************************************************************************/
	GPU_ONLY void CopyFromVector(const real * const values, size_type count);
};

} } } } // namespace axis::foundation::blas

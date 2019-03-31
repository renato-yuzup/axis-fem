/**********************************************************************************************//**
 * @file	foundation/blas/DenseMatrix.hpp
 *
 * @brief	Declares the class of a generic matrix and 
 * 			a second-order tensor.
 **************************************************************************************************/
#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* @brief	A regular matrix with no specific properties.
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @sa	self
	**************************************************************************************************/
class DenseMatrix
{
public:

	/**********************************************************************************************//**
		* @brief	Defines an alias representing the type of this object.
		**************************************************************************************************/
	typedef DenseMatrix self;

	/**********************************************************************************************//**
		* @brief	Default copy constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other object from which data will be copied.
		**************************************************************************************************/
	GPU_ONLY DenseMatrix(const self& other);

	/**********************************************************************************************//**
		* @brief	Creates a new matrix with the specified dimensions.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	rows   	Number of rows for this matrix.
		* @param	columns	Number of columns for this matrix.
		**************************************************************************************************/
	GPU_ONLY DenseMatrix(size_type rows, size_type columns);

	/**********************************************************************************************//**
		* @brief	Creates a new matrix and initializes all elements with a
		* 			given value.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	rows			Number of rows for this matrix.
		* @param	columns			Number of columns for this matrix.
		* @param	initialValue	The value to be assigned to all elements
		* 							in the matrix.
		**************************************************************************************************/
	GPU_ONLY DenseMatrix(size_type rows, size_type columns, real initialValue);

	/**********************************************************************************************//**
		* @brief	Creates a new matrix and initializes all elements with
		* 			the values in an array.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	rows   	Number of rows for this matrix.
		* @param	columns	Number of columns for this matrix.
		* @param	values 	Pointer to an array containing the values to
		* 					which matrix elements will be initialized. Values
		* 					are stored in row-major order contiguously. The
		* 					array is supposed to be at least the same size as
		* 					the element count of the matrix.
		**************************************************************************************************/
	GPU_ONLY DenseMatrix(size_type rows, size_type columns, const real * const values);

	/**********************************************************************************************//**
		* @brief	Creates a new matrix and initializes the \em n first
		* 			elements with the values in an array. Remaining positions
		* 			will be initialized to zero.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	rows			Number of rows for this matrix.
		* @param	columns			Number of columns for this matrix.
		* @param	values			Pointer to the array containing the
		* 							initialization values.
		* @param	elementCount	Tells how many elements ordered in a row-
		* 							major fashion will be initialized. It is
		* 							supposed the the initialization array size is
		* 							at least of this length.
		**************************************************************************************************/
	GPU_ONLY DenseMatrix(size_type rows, size_type columns, const real * const values, size_type elementCount);

	/**********************************************************************************************//**
		* @brief	Destructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY ~DenseMatrix(void);

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
	GPU_ONLY real operator ()(size_type row, size_type column) const;

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
	GPU_ONLY real& operator ()(size_type row, size_type column);

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
		* 				fulfill this operation. Some implications might exist
		* 				for each subclass that implement this method. However,
		* 				it is guaranteed that the data is copied to this matrix,
		* 				assuming restrictions applied by the subclass, though.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	source	self from which to copy all data.
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
		* @brief	Stores in this object the result of A + B, where A is this matrix and B is another matrix.
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
		* @brief	Stores in this object the result of A - B, where A is this matrix and B is another matrix.
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

	size_type _rows, _columns;
	real * _data;
  axis::yuzu::foundation::memory::RelativePointer _ptr;
  int sourceMemory_;


	/**********************************************************************************************//**
		* @brief	Initialises the matrix object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	rows   	          Row count for this matrix.
		* @param	columns	          Column count for this matrix.
    * @param	allocationSource	Tells from where memory should be requested.
		**************************************************************************************************/
	GPU_ONLY void Init(size_type rows, size_type columns);

	/**********************************************************************************************//**
		* @brief	Copies data from an array.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	The array of values from which data will be
		* 					copied.
		* @param	count 	Number of elements to be copied.
		**************************************************************************************************/
	GPU_ONLY void CopyFromVector(const real * const values, size_type count);
};
		
/**********************************************************************************************//**
	* @brief	Defines an alias for a second-order tensor, which is
	* 			represented by a dense matrix.
	**************************************************************************************************/
typedef DenseMatrix SecondTensor;

} } } } // namespace axis::yuzu::foundation::blas


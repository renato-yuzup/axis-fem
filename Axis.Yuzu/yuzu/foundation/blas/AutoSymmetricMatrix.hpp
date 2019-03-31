/**********************************************************************************************//**
 * @file	foundation\blas\AutoSymmetricMatrix.hpp
 *
 * @brief	Declares the symmetric matrix class.
 **************************************************************************************************/
#pragma once
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* @class	AutoSymmetricMatrix
	*
	* @brief	A square matrix which is equal to its transpose.
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @sa	Matrix
	**************************************************************************************************/
template <int N>
class AutoSymmetricMatrix
{
public:

	/**********************************************************************************************//**
		* @brief	Defines an alias for the type of this object.
		**************************************************************************************************/
	typedef AutoSymmetricMatrix<N> self;

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY AutoSymmetricMatrix(void)
  {
    // nothing to do here
  }

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix and sets all values to a given constant.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	value	The value to assign to each matrix element.
		**************************************************************************************************/
	explicit GPU_ONLY AutoSymmetricMatrix(real value)
  {
    SetAll(value);
  }

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix and initializes all values
		* 			as specified in an initialization array.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	The initialization array containing the values in
		* 					a row-major order for the lower symmetric part of
		* 					the matrix.
		**************************************************************************************************/
	explicit GPU_ONLY AutoSymmetricMatrix(const real * const values)
  {
    CopyFromVector(values, N*(N+1)/2);
  }

	/**********************************************************************************************//**
		* @brief	Creates a new symmetric matrix and initializes some
		* 			values as specified in an initialization array. Remaining
		* 			values are initialized to zero.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	The initialization values in a row-major order
		* 					for the lower symmetric part of the matrix.
		* @param	count 	Number of elements to be initialized by the
		* 					initialization array, starting from index (0,0).
		**************************************************************************************************/
	GPU_ONLY AutoSymmetricMatrix(const real * const values, size_type count)
  {
    CopyFromVector(values, count);
  }

	/**********************************************************************************************//**
		* @brief	Copy constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other matrix.
		**************************************************************************************************/
	GPU_ONLY AutoSymmetricMatrix(const self& other)
  {
    CopyFrom(other);
  }

	/**********************************************************************************************//**
		* @brief	Default destructor for this class.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY ~AutoSymmetricMatrix(void)
  {
    // nothing to do here
  }

	/**********************************************************************************************//**
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY void Destroy(void) const
  {
    // nothing to do here
  }

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
	GPU_ONLY real& operator ()(size_type row, size_type column)
  {
    size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
    return data_[pos];
  }

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
	GPU_ONLY real operator ()(size_type row, size_type column) const
  {
    size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
    return data_[pos];
  }

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
	GPU_ONLY real GetElement(size_type row, size_type column) const
  {
    size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
    return data_[pos];
  }

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
	GPU_ONLY self& SetElement(size_type row, size_type column, real value)
  {
    size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
    data_[pos] = value;
    return *this;
  }

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
	GPU_ONLY self& Accumulate(size_type row, size_type column, real value)
  {
    size_type pos = (column > row)? (column+1)*column/2 + row : (row+1)*row/2 + column;
    data_[pos] += value;
    return *this;
  }

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
	GPU_ONLY self& CopyFrom(const self& source)
  {
    for (int i = 0; i < N; i++)
    {
      for (int j = i; j < N; j++)
      {
        operator()(i,j) = source(i,j);
      }
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Assigns zeros to all elements of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	This matrix object.
		**************************************************************************************************/
	GPU_ONLY self& ClearAll(void)
  {
    return SetAll(0);
  }

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
	GPU_ONLY self& Scale(real value)
  {
    for (int i = 0; i < N*(N+1)/2; i++)
    {
      data_[i] *= value;
    }
    return *this;
  }

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
	GPU_ONLY self& SetAll(real value)
  {
    for (int i = 0; i < N*(N+1)/2; i++)
    {
      data_[i] = value;
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Returns the number of rows of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-null, positive number representing the number of
		* 			rows.
		**************************************************************************************************/
	GPU_ONLY size_type Rows(void) const
  {
    return N;
  }

	/**********************************************************************************************//**
		* @brief	Returns the number of columns of this matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-null, positive number representing the number of
		* 			columns.
		**************************************************************************************************/
	GPU_ONLY size_type Columns(void) const
  {
    return N;
  }

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
	GPU_ONLY size_type ElementCount(void) const
  {
    return N*N;
  }

	/**********************************************************************************************//**
		* @brief	Returns the number of elements in the lower symmetric
		* 			part (including the main diagonal) of the matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type SymmetricCount(void) const
  {
    return N*(N+1)/2;
  }

	/**********************************************************************************************//**
		* @brief	Tells if the column and row count are the same.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	true is both row and column count are the same, false
		* 			otherwise.
		**************************************************************************************************/
	GPU_ONLY bool IsSquare(void) const
  {
    return true;
  }

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
	GPU_ONLY self& operator = (const self& other)
  {
    return CopyFrom(other);
  }

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
	GPU_ONLY self& operator += (const self& other)
  {
    for (int i = 0; i < N; i++)
    {
      for (int j = i; j < N; j++)
      {
        operator()(i,j) += other(i,j);
      }
    }
    return *this;
  }

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
	GPU_ONLY self& operator -= (const self& other)
  {
    for (int i = 0; i < N; i++)
    {
      for (int j = i; j < N; j++)
      {
        operator()(i,j) -= other(i,j);
      }
    }
    return *this;
  }

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
	GPU_ONLY self& operator *= (real scalar)
  {
    return Scale(scalar);
  }

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
	GPU_ONLY self& operator /= (real scalar)
  {
    for (int i = 0; i < N*(N+1)/2; i++)
    {
      data_[i] /= scalar;
    }
    return *this;
  }

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
	GPU_ONLY real Trace(void) const
  {
    real trace = 0;
    for (int i = 0; i < N; i++)
    {
      trace += operator()(i,i);
    }
    return trace;
  }
private:
	real data_[N*(N+1)/2];

	/**********************************************************************************************//**
		* @brief	Initializes this object and copies values from an initialization vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	Pointer to the array of initialization values.
		* @param	count 	Number of elements in the symmetric part.
		**************************************************************************************************/
	GPU_ONLY void CopyFromVector(const real * const values, size_type count)
  {
    for (size_type i = 0; i < count; ++i)
    {
      data_[i] = values[i];
    }

    // fill unassigned spaces with zeros
    for (size_type i = count; i < N*(N+1)/2; ++i)
    {
      data_[i] = 0;
    }
  }
};

} } } } // namespace axis::yuzu::foundation::blas


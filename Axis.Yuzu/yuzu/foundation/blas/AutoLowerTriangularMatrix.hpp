/**********************************************************************************************//**
 * @file	foundation/blas/AutoLowerTriangularMatrix.hpp
 *
 * @brief	Declares the triangular matrix class.
 **************************************************************************************************/
#pragma once
#include "yuzu/common/gpu.hpp"

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
template <int N>
class AutoLowerTriangularMatrix
{
public:

	/**********************************************************************************************//**
		* @brief	Defines an alias for the type of this object.
		**************************************************************************************************/
	typedef AutoLowerTriangularMatrix<N> self;

	/**********************************************************************************************//**
		* @brief	Copy constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other matrix.
		**************************************************************************************************/
	GPU_ONLY AutoLowerTriangularMatrix(const self& other)
  {
    CopyFrom(other);
  }

	/**********************************************************************************************//**
		* @brief	Creates a new triangular matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY AutoLowerTriangularMatrix(void)
  {
    // nothing to do here
  }

	/**********************************************************************************************//**
		* @brief	Creates a new triangular matrix and assigns to every
		* 			element the specified constant.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	value	The value to be assigned to the matrix elements.
		**************************************************************************************************/
  explicit GPU_ONLY AutoLowerTriangularMatrix(real value)
  {
    SetAll(value);
  }

  explicit GPU_ONLY AutoLowerTriangularMatrix(const real * const values)
  {
    CopyFromVector(values, N*(N+1)/2);
  }

	GPU_ONLY AutoLowerTriangularMatrix(const real * const values, size_type count)
  {
    CopyFromVector(values, count);
  }

	/**********************************************************************************************//**
		* @brief	Default destructor for this class.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY ~AutoLowerTriangularMatrix(void)
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
    return data_[row * (row+1) / 2 + column];
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
    return (row >= column)? data_[row * (row+1) / 2 + column] : 0;
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
    return (row >= column)? data_[row * (row+1) / 2 + column] : 0;
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
    data_[row * (row+1) / 2 + column] = value;
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
    data_[row * (row+1) / 2 + column] += value;
    return *this;
  }

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
	GPU_ONLY self& CopyFrom(const self& source)
  {
    for (int row = 0; row < N; row++)
    {
      for (int col = 0; col < N; col++)
      {
        data_[row * (row+1) / 2 + column] = source(row, col);
      }
    }
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
		* @brief	Returns the number of writable elements in this matrix,
		* 			that is, the number of elements in lower/upper triangular
		* 			part (including main diagonal) of the matrix.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type TriangularCount(void) const
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
    for (int row = 0; row < N; row++)
    {
      for (int col = 0; col < N; col++)
      {
        data_[row * (row+1) / 2 + column] += other(row, col);
      }
    }
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
    for (int row = 0; row < N; row++)
    {
      for (int col = 0; col < N; col++)
      {
        data_[row * (row+1) / 2 + column] -= other(row, col);
      }
    }
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
    real t = 0;
    for (size_type i = 0; i < N; ++i)
    {
      t += operator()(i,i);
    }
    return t;
  }
private:
	real data_[N*(N+1) / 2];

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
	GPU_ONLY void CopyFromVector(const real * const values, size_type count)
  {
    for (size_type i = 0; i < count; ++i)
    {
      data_[i] = values[i];
    }

    // fill unassigned spaces with zeros
    for (size_type i = count; i < N*(N+1) / 2; ++i)
    {
      data_[i] = 0;
    }
  }
};

} } } } // namespace axis::foundation::blas

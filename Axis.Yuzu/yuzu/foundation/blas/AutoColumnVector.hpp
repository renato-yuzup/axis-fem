/**********************************************************************************************//**
 * @file	foundation/blas/AutoColumnVector.hpp
 *
 * @brief	Contains definitions of classes for vector manipulation.
 **************************************************************************************************/
#pragma once
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* @brief	Defines a unidimensional matrix (that is, a vector).
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @sa	Vector
	**************************************************************************************************/
template <int N>
class AutoColumnVector 
{
public:

	/**********************************************************************************************//**
		* @brief	Defines an alias for the type of this object.
		**************************************************************************************************/
	typedef AutoColumnVector<N> self;

	/**********************************************************************************************//**
		* @brief	Creates a new vector.
		**************************************************************************************************/
	GPU_ONLY AutoColumnVector(void)
  {
    // nothing to do here
  }

	/**********************************************************************************************//**
		* @brief	Copy constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other vector.
		**************************************************************************************************/
	GPU_ONLY AutoColumnVector(const self& other)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] = other(i);
    }
  }

	/**********************************************************************************************//**
		* @brief	Creates a new vector and sets
		* 			all positions to an initialization value.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	initialValue	The initialization value.
		**************************************************************************************************/
	explicit GPU_ONLY AutoColumnVector(real initialValue)
  {
    SetAll(initialValue);
  }

	/**********************************************************************************************//**
		* @brief	Creates a new vector and sets
		* 			all values according to a given initialization array.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	Pointer to the initialization array containing
		* 					the initialization values for the vector elements.
		**************************************************************************************************/
	GPU_ONLY AutoColumnVector(const real * const values)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] = values[i];
    }
  }

	/**********************************************************************************************//**
		* @brief	Creates a new vector and sets
		* 			some elements to a given array of initialization values.
		* 			Other positions are initialized to zero.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values			Pointer to the array of initialization
		* 							values.
		* @param	elementCount	Number of elements to be initialized from
		* 							the array. Elements starting from index 0 of
		* 							the vector will be initialized.
		**************************************************************************************************/
	GPU_ONLY AutoColumnVector(const real * const values, size_type elementCount)
  {
    for (int i = 0; i < elementCount; i++)
    {
      data_[i] = values[i];
    }
    for (int i = elementCount; i < N; i++)
    {
      data_[i] = 0;
    }
  }

	/**********************************************************************************************//**
		* @brief	Default destructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY ~AutoColumnVector(void)
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
		* @brief	Return a vector element in the specified position.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	pos		The zero-based index of the element inside the vector.
		* @return	The vector element value.
		**************************************************************************************************/
	GPU_ONLY real operator ()(size_type pos) const
  {
    return data_[pos];
  }

	/**********************************************************************************************//**
		* @brief	Return a vector element in the specified position.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	pos		The zero-based index of the element inside the vector.
		* @return	A writable reference to the vector element.
		**************************************************************************************************/
	GPU_ONLY real& operator ()(size_type pos)
  {
    return data_[pos];
  }

	/**********************************************************************************************//**
		* @brief	Gets a value stored in a vector element.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	pos	The zero-based index of the element.
		*
		* @return	The value stored in the specified element.
		**************************************************************************************************/
	GPU_ONLY real GetElement(size_type pos) const
  {
    return data_[pos];
  }

	/**********************************************************************************************//**
		* @brief	Assigns a value to an element in the vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	pos  	The zero-based index of the element. 
		* @param	value	The value to be assigned.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& SetElement(size_type pos, real value)
  {
    data_[pos] = value;
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Increments the value stored in a vector element by a 
		* 			specified amount.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	pos  	The zero-based index of the element.
		* @param	value	The amount by which the element value will be incremented.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& Accumulate(size_type pos, real value)
  {
    data_[pos] += value;
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Copies the contents from another vector to this object.
		* 			@note	The source object must be of the same dimensions as
		* 			this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	source	The vector from which element values will be
		* 					copied.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& CopyFrom(const self& source)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] = source(i);
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Sets all vector elements to zero.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& ClearAll(void)
  {
    SetAll(0);
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Multiplies every vector element by a given factor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	value	The factor by which each element value will be multiplied.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& Scale(real value)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] *= value;
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Sets all vector elements to a given value.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	value	The value to which every element value will be assigned.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& SetAll(real value)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] = value;
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Returns the element count of this vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type Length(void) const
  {
    return N;
  }

	/**********************************************************************************************//**
		* @brief	Copies the contents from another vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other vector.
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& operator = (const self& other)
  {
    return CopyFrom(other);
  }

	/**********************************************************************************************//**
		* @brief	Stores in this object the result of A + B, where A is
		* 			this vector and B is another vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other vector (B).
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& operator += (const self& other)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] += other(i);
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Stores in this object the result of A - B, where A is
		* 			this vector and B is another vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other vector (B).
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& operator -= (const self& other)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] -= other(i);
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Multiplies every element value by a given factor.
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
    for (int i = 0; i < N; i++)
    {
      data_[i] *= scalar;
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Divides every element value by a given divisor.
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
    for (int i = 0; i < N; i++)
    {
      data_[i] /= scalar;
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Returns the scalar product of the transpose of this
		* 			vector by itself or the scalar product of itself and its
		* 			transpose, whichever is suitable.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A real number representing the scalar product.
		**************************************************************************************************/
	GPU_ONLY real SelfScalarProduct(void) const
  {
    real scalar = 0;
    for (int i = 0; i < N; i++)
    {
      scalar += data_[i] * data_[i];
    }
    return scalar;
  }

	/**********************************************************************************************//**
		* @brief	Returns the norm of this vector, which is the square root
		* 			of its self scalar product.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A real number representing the norm.
		**************************************************************************************************/
	GPU_ONLY real Norm(void) const
  {
    return sqrt(SelfScalarProduct());
  }

	/**********************************************************************************************//**
		* @brief	Gets all vector elements value to its inverse, that is,
		* 			<em>1/e</em>, where \em e is the current element value.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& Invert(void)
  {
    for (int i = 0; i < N; i++)
    {
      data_[i] = 1.0 / data_[i];
    }
    return *this;
  }

	/**********************************************************************************************//**
		* @brief	Returns the row count of this vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type Rows(void) const
  {
    return length;
  }

	/**********************************************************************************************//**
		* @brief	Returns the column count of this vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type Columns(void) const
  {
    return 1;
  }

	/**********************************************************************************************//**
		* @brief	Compares if this vector is equal to another.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	true if all elements in both vectors are equal. Note that dimensions must be the same.
    *         Floating-point roundoff errors might trigger false conditions on comparison.
		**************************************************************************************************/
	GPU_ONLY bool operator ==(const self& other) const
  {
    bool equal = true;
    for (int i = 0; i < N && equal; i++)
    {
      equal = data_[i] == other(i);
    }
    return equal;
  }

	/**********************************************************************************************//**
		* @brief	Compares if this vector is different from another.
		*
		* @return	true if all elements in both vectors are different. Note that dimensions must be the same.
    *         Floating-point roundoff errors might trigger false conditions on comparison.
		**************************************************************************************************/
  GPU_ONLY bool operator !=(const self& other) const
  {
    return !(*this == other);
  }

private:
  real data_[N];
};


} } } } // namespace axis::yuzu::foundation::blas

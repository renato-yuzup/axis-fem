/**********************************************************************************************//**
 * @file	foundation/blas/RowVector.hpp
 *
 * @brief	Contains definitions of classes for vector manipulation.
 **************************************************************************************************/
#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* @brief	Defines a unidimensional matrix (that is, a vector).
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @sa	Vector
	**************************************************************************************************/
class RowVector 
{
public:

	/**********************************************************************************************//**
		* @brief	Defines an alias for the type of this object.
		**************************************************************************************************/
	typedef RowVector self;

	/**********************************************************************************************//**
		* @brief	Copy constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	other	The other vector.
		**************************************************************************************************/
	GPU_ONLY RowVector(const self& other);

	/**********************************************************************************************//**
		* @brief	Creates a new vector with the specified length.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	length	The vector length.
		**************************************************************************************************/
	explicit GPU_ONLY RowVector(size_type length);

	/**********************************************************************************************//**
		* @brief	Creates a new vector with the specified length and sets
		* 			all positions to an initialization value.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	length			The vector length.
		* @param	initialValue	The initialization value.
		**************************************************************************************************/
	GPU_ONLY RowVector(size_type length, real initialValue);

	/**********************************************************************************************//**
		* @brief	Creates a new vector with the specified length and sets
		* 			all values according to a given initialization array.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	length	The vector length.
		* @param	values	Pointer to the initialization array containing
		* 					the initialization values for the vector elements.
		**************************************************************************************************/
	GPU_ONLY RowVector(size_type length, const real * const values);

	/**********************************************************************************************//**
		* @brief	Creates a new vector with the specified length and sets
		* 			some elements to a given array of initialization values.
		* 			Other positions are initialized to zero.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	length			The vector length.
		* @param	values			Pointer to the array of initialization
		* 							values.
		* @param	elementCount	Number of elements to be initialized from
		* 							the array. Elements starting from index 0 of
		* 							the vector will be initialized.
		**************************************************************************************************/
	GPU_ONLY RowVector(size_type length, const real * const values, size_type elementCount);

	/**********************************************************************************************//**
		* @brief	Default destructor.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY ~RowVector(void);

	/**********************************************************************************************//**
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		**************************************************************************************************/
	GPU_ONLY void Destroy(void) const;

	/**********************************************************************************************//**
		* @brief	Return a vector element in the specified position.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	pos		The zero-based index of the element inside the vector.
		* @return	The vector element value.
		**************************************************************************************************/
	GPU_ONLY real operator ()(size_type pos) const;

	/**********************************************************************************************//**
		* @brief	Return a vector element in the specified position.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	pos		The zero-based index of the element inside the vector.
		* @return	A writable reference to the vector element.
		**************************************************************************************************/
	GPU_ONLY real& operator ()(size_type pos);

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
	GPU_ONLY real GetElement(size_type pos) const;

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
	GPU_ONLY self& SetElement(size_type pos, real value);

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
	GPU_ONLY self& Accumulate(size_type pos, real value);

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
	GPU_ONLY self& CopyFrom(const self& source);

	/**********************************************************************************************//**
		* @brief	Sets all vector elements to zero.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& ClearAll(void);

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
	GPU_ONLY self& Scale(real value);

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
	GPU_ONLY self& SetAll(real value);

	/**********************************************************************************************//**
		* @brief	Returns the element count of this vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type Length(void) const;

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
	GPU_ONLY self& operator = (const self& other);

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
	GPU_ONLY self& operator += (const self& other);

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
	GPU_ONLY self& operator -= (const self& other);

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
	GPU_ONLY self& operator *= (real scalar);

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
	GPU_ONLY self& operator /= (real scalar);

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
	GPU_ONLY real SelfScalarProduct(void) const;

	/**********************************************************************************************//**
		* @brief	Returns the norm of this vector, which is the square root
		* 			of its self scalar product.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A real number representing the norm.
		**************************************************************************************************/
	GPU_ONLY real Norm(void) const;

	/**********************************************************************************************//**
		* @brief	Gets all vector elements value to its inverse, that is,
		* 			<em>1/e</em>, where \em e is the current element value.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A reference to this object.
		**************************************************************************************************/
	GPU_ONLY self& Invert(void);

	/**********************************************************************************************//**
		* @brief	Returns the row count of this vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type Rows(void) const;

	/**********************************************************************************************//**
		* @brief	Returns the column count of this vector.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	A non-zero, positive integer number.
		**************************************************************************************************/
	GPU_ONLY size_type Columns(void) const;

	/**********************************************************************************************//**
		* @brief	Compares if this vector is equal to another.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @return	true if all elements in both vectors are equal. Note that dimensions must be the same.
    *         Floating-point roundoff errors might trigger false conditions on comparison.
		**************************************************************************************************/
	GPU_ONLY bool operator ==(const self& other) const;

	/**********************************************************************************************//**
		* @brief	Compares if this vector is different from another.
		*
		* @return	true if all elements in both vectors are different. Note that dimensions must be the same.
    *         Floating-point roundoff errors might trigger false conditions on comparison.
		**************************************************************************************************/
  GPU_ONLY bool operator !=(const self& other) const;
private:
	real * _data;
  axis::yuzu::foundation::memory::RelativePointer _ptr;
  int sourceMemory_;
	size_type _length;

	/**********************************************************************************************//**
		* @brief	Initializes this object.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	length	      Vector length.
		**************************************************************************************************/
	GPU_ONLY void Init(size_type length);

	/**********************************************************************************************//**
		* @brief	Initializes this object and sets its values according to an array.
		*
		* @author	Renato T. Yamassaki
		* @date	19 ago 2012
		*
		* @param	values	Pointer to the array containing the initialization values.
		* @param	count 	Number elements to be initialized.
		**************************************************************************************************/
	GPU_ONLY void CopyFromVector(const real * const values, size_type count);
};

} } } } // namespace axis::yuzu::foundation::blas
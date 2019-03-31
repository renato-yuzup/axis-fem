#pragma once
#include <math.h>
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* <summary> Returns a read-only 3x3 identity matrix.</summary>
	*
	* <returns> A reference to the identity matrix.</returns>
	**************************************************************************************************/
template<class Matrix>
GPU_ONLY void Identity3D(Matrix& matrix)
{
  matrix.ClearAll();
  matrix(0,0) = 1; matrix(1,1) = 1; matrix(2,2) = 1; 
}

/**********************************************************************************************//**
* @brief	Sets all matrix elements to zero, except for the main
* 			diagonal elements which are assigned the unity value (one).
*
* @author	Renato T. Yamassaki
* @date	19 ago 2012
*
* @param [in,out]	eye	The matrix to be modified.
**************************************************************************************************/
template <class Matrix>
GPU_ONLY void Identity(Matrix& eye)
{
	eye.ClearAll();

	size_type n = eye.Rows();
	for (size_type i = 0; i < n; ++i)
	{
		eye(i,i) = 1;
	}
}

/**********************************************************************************************//**
* @brief	Returns the value of the Kronecker delta function for a
* 			specified pair of indices.
*
* @author	Renato T. Yamassaki
* @date	19 ago 2012
*
* @param	i	First index.
* @param	j	Second index.
*
* @return	Returns 1 (one) if, and only if, indices are equal, zero
* 			otherwise..
**************************************************************************************************/
inline GPU_ONLY real KroneckerDelta(size_type i, size_type j)
{
  return ((i == j)? 1 : 0);
}

/**************************************************************************************************
	* <summary>	Calculates the distance between two points in a three-dimensional space. </summary>
	*
	* <param name="x1">	The first point x value. </param>
	* <param name="y1">	The first point y value. </param>
	* <param name="z1">	The first point z value. </param>
	* <param name="x2">	The second point x value. </param>
	* <param name="y2">	The second point y value. </param>
	* <param name="z2">	The second point z value. </param>
	*
	* <returns>	The distance between the points. </returns>
	**************************************************************************************************/
inline GPU_ONLY real Distance3D(real x1, real y1, real z1, real x2, real y2, real z2)
{
  real x = x1 - x2;
  real y = y1 - y2;
  real z = z1 - z2;
  return sqrt(x*x + y*y + z*z);
}

/**************************************************************************************************
	* <summary>	Calculates the determinant of a 3x3 matrix.
	*
	* <param name="matrix">	The matrix. </param>
	*
	* <returns>	The determinant. </returns>
	**************************************************************************************************/
template<class Matrix>
inline GPU_ONLY real Determinant3D(const Matrix& matrix)
{
  const Matrix& m = matrix;
  return m(0,0)*m(1,1)*m(2,2) - m(0,0)*m(1,2)*m(2,1) - m(0,1)*m(1,0)*m(2,2) + 
    m(0,1)*m(1,2)*m(2,0) + m(0,2)*m(1,0)*m(2,1) - m(0,2)*m(1,1)*m(2,0);
}

} } } } // namespace axis::yuzu::foundation::blas

/**********************************************************************************************//**
* @file	foundation/blas/matrix_compatibility.hpp
*
* @brief	Declares functions for checking dimension compatibility 
* 			in matrix operations.
**************************************************************************************************/
#pragma once
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* @brief	Checks if two matrix are dimension-compatible for matrix
	* 			product operations.
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @param	m1				The first (left) matrix.
	* @param	m2				The second (right) matrix.
	* @param	m1Transposed	(optional) tells if the transpose of the
	* 							first matrix should be used instead.
	* @param	m2Transposed	(optional) tells if the transpose of the
	* 							second matrix should be used instead.
	*
	* @return	true if the pair is compatible, false otherwise.
	**************************************************************************************************/
template <class Matrix>
GPU_ONLY bool MatrixIsProductCompatible(const Matrix& m1, const Matrix& m2, bool m1Transposed = false, bool m2Transposed = false)
{
  size_type c1, r2;
  c1 = m1Transposed? m1.Rows() : m1.Columns();
  r2 = m2Transposed? m2.Columns() : m2.Rows();
  return (c1 == r2);
}

/**********************************************************************************************//**
	* @brief	Checks if a matrix and a vector are dimension-compatible for matrix-vector
	* 			product operations.
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @param	m1				The left matrix.
	* @param	m2				The right vector.
	* @param	m1Transposed	(optional) tells if the transpose of the
	* 							matrix should be used instead.
	* @param	m2Transposed	(optional) tells if the transpose of the
	* 							vector should be used instead.
	*
	* @return	true if the pair is compatible, false otherwise.
	**************************************************************************************************/
template <class T1, class T2>
GPU_ONLY bool MatrixIsProductCompatible(const T1& m1, const T2& m2, bool m1Transposed = false, bool m2Transposed = false)
{
  size_type c1, r2;
  c1 = m1Transposed? m1.Rows() : m1.Columns();
  r2 = m2Transposed? m2.Columns() : m2.Rows();
  return (c1 == r2);
}

/**********************************************************************************************//**
	* @brief	Checks if a matrix has the same dimensions as specified.
	*
	* @author	Renato T. Yamassaki
	* @date	19 ago 2012
	*
	* @param	m			   	The matrix to be analyzed.
	* @param	rows		   	The number of rows.
	* @param	columns		   	The number of columns.
	* @param	storeTransposed	(optional) tells to evaluate using the transpose of the matrix
	*
	* @return	true if the matrix is compatible, false otherwise.
	**************************************************************************************************/
template <class Matrix>
GPU_ONLY bool MatrixIsResultCompatible(const Matrix& m, size_type rows, size_type columns, bool storeTransposed = false)
{
  return storeTransposed? (m.Rows() == columns && m.Columns() == rows) : (m.Rows() == rows && m.Columns() == columns);
}

} } } } // namespace axis::yuzu::foundation::blas

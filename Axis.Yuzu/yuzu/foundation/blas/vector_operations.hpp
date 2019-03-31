#pragma once
#include "yuzu/common/gpu.hpp"
#include "blas_basic_types.hpp"
#include "matrix_compatibility.hpp"
#include "DenseMatrix.hpp"
#include "SymmetricMatrix.hpp"
#include "AutoDenseMatrix.hpp"
#include "AutoSymmetricMatrix.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**************************************************************************************************
	* <summary>	Calculates the scalar (or dot) product between two vectors. </summary>
	*
	* <param name="v1">	The first (left) vector. </param>
	* <param name="v2">	The second (right) vector. </param>
	*
	* <returns>	A scalar representing the dot product of the vectors. </returns>
	**************************************************************************************************/
template <class V1, class V2>
GPU_ONLY real VectorScalarProduct(const V1& v1, const V2& v2)
{
  size_type len = v1.Length();
  real result = 0;
  for (size_type i = 0; i < len; ++i)
  {
    result += v1(i)*v2(i);
  }
  return result;
}

/**************************************************************************************************
	* <summary>	Stores in the LHS vector the product of a scalar and another vector. </summary>
	*
	* <param name="lhs">   	[in,out] The left hand side vector. </param>
	* <param name="scalar">	The scalar. </param>
	* <param name="v">			The vector. </param>
	**************************************************************************************************/
template <class V1, class V2>
GPU_ONLY void VectorScale(V1& lhs, real scalar, const V2& v)
{
  size_type len = v.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) = scalar*v(i);
  }
}

/**************************************************************************************************
	* <summary>	Swaps contents of two vectors. </summary>
	*
	* <param name="v1">	[in,out] The first vector. </param>
	* <param name="v2">	[in,out] The second vector. </param>
	**************************************************************************************************/
template <class V>
GPU_ONLY void VectorSwap(V& v1, V& v2)
{
  size_type len = v1.Length();
  for (size_type i = 0; i < len; ++i)
  {
    real val = v1(i);
    v1(i) = v2(i);
    v2(i) = val;
  }
}

/**************************************************************************************************
	* <summary>	Stores in the LHS vector the sum of two vectors. </summary>
	*
	* <param name="lhs">	  	The left hand side vector. </param>
	* <param name="v1">	  	The first vector. </param>
	* <param name="v2Scalar">	The scalar by which the second vector parcel will be multiplied. </param>
	* <param name="v2">	  	The second vector. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorSum(V1& lhs, const V2& v1, real v2Scalar, const V3& v2)
{
  size_type len = v1.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) = v1(i) + v2Scalar*v2(i);
  }
}

/**************************************************************************************************
* <summary>	Stores in the LHS vector the sum of two vectors. </summary>
	*
	* <param name="lhs">	  	The left hand side vector. </param>
	* <param name="v1Scalar">	The scalar by which the first vector parcel will be multiplied. </param>
	* <param name="v1">	  	The first vector. </param>
	* <param name="v2Scalar">	The scalar by which the second vector parcel will be multiplied. </param>
	* <param name="v2">	  	The second vector. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorSum(V1& lhs, real v1Scalar, const V2& v1, real v2Scalar, const V3& v2)
{
  size_type len = v1.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) = v1Scalar*v1(i) + v2Scalar*v2(i);
  }
}

/**************************************************************************************************
	* <summary>	Accumulates in the LHS vector the sum of two vectors. </summary>
	*
	* <param name="lhs">	  	The left hand side vector. </param>
	* <param name="v1">	  	The first vector. </param>
	* <param name="v2Scalar">	The scalar by which the second vector parcel will be multiplied. </param>
	* <param name="v2">	  	The second vector. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorAccumulateSum(V1& lhs, const V2& v1, real v2Scalar, const V3& v2)
{
  size_type len = v1.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) += (v1(i) + v2Scalar*v2(i));
  }
}

/**************************************************************************************************
* <summary>	Accumulates in the LHS vector the sum of two vectors. </summary>
	*
	* <param name="lhs">	  	The left hand side vector. </param>
	* <param name="v1Scalar">	The scalar by which the first vector parcel will be multiplied. </param>
	* <param name="v1">	  	The first vector. </param>
	* <param name="v2Scalar">	The scalar by which the second vector parcel will be multiplied. </param>
	* <param name="v2">	  	The second vector. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorAccumulateSum(V1& lhs, real v1Scalar, const V2& v1, real v2Scalar, const V3& v2)
{
  size_type len = v1.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) += (v1Scalar*v1(i) + v2Scalar*v2(i));
  }
}

/**************************************************************************************************
	* <summary>	Stores in the LHS vector the element-by-element product of two vectors.</summary>
	*
	* <param name="lhs">   	[in,out] The left hand side vector. </param>
	* <param name="scalar">	The scalar which each product will be multiplied. </param>
	* <param name="v1">		The first vector factor. </param>
	* <param name="v2">		The second vector factor. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorElementProduct(V1& lhs, real scalar, const V2& v1, const V3& v2)
{
  size_type len = v1.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) = scalar * v1(i) * v2(i);
  }
}

/**************************************************************************************************
	* <summary>	Stores in the LHS vector the element-by-element division of two vectors.</summary>
	*
	* <param name="lhs">		 	The left hand side vector. </param>
	* <param name="scalar">	 	The scalar by which each quotient will be multiplied. </param>
	* <param name="numerator">  	The vector which contains the dividends. </param>
	* <param name="denominator">	The vector which contains the divisors. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorElementDivide(V1& lhs, real scalar, const V2& numerator, const V3& denominator)
{
  size_type len = numerator.Length();
	for (size_type i = 0; i < len; ++i)
	{
		lhs(i) = scalar * numerator(i) / denominator(i);
	}
}

/**************************************************************************************************
	* <summary>	Accumulates in the LHS vector the element-by-element product of two vectors.</summary>
	*
	* <param name="lhs">   	[in,out] The left hand side vector. </param>
	* <param name="scalar">	The scalar which each product will be multiplied. </param>
	* <param name="v1">		The first vector factor. </param>
	* <param name="v2">		The second vector factor. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorAccumulateElementProduct(V1& lhs, real scalar, const V2& v1, const V3& v2)
{
  size_type len = v1.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) += (scalar * v1(i) * v2(i));
  }
}

/**************************************************************************************************
	* <summary>	Stores in the LHS vector the element-by-element division of two vectors.</summary>
	*
	* <param name="lhs">		 	The left hand side vector. </param>
	* <param name="scalar">	 	The scalar by which each quotient will be multiplied. </param>
	* <param name="numerator">  	The vector which contains the dividends. </param>
	* <param name="denominator">	The vector which contains the divisors. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorAccumulateElementDivide(V1& lhs, real scalar, const V2& numerator, const V3& denominator)
{
  size_type len = numerator.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) += (scalar * numerator(i) / denominator(i));
  }
}

/**********************************************************************************************//**
	* <summary> Solves the linear system formed with a diagonal
	* 			 coefficient matrix and a RHS vector, that is, calculates
	* 			 the result of M^(-1) * v (scalars omitted, refer to
	* 			 parameter list for more information). Finally, results
	* 			 are stored in the LHS vector.</summary>
	*
	* <param name="lhs">			    The left hand side vector.</param>
	* <param name="coefficientScalar"> The scalar which multiplies the
	* 									diagonal coefficient matrix
	* 									before inversion.</param>
	* <param name="diagonalMatrix">    The diagonal coefficient matrix
	* 									represented as a vector.</param>
	* <param name="vScalar">		    The scalar which multiplies the
	* 									RHS vector.</param>
	* <param name="v">				    The RHS vector.</param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorSolve(V1& lhs, real coefficientScalar, const V2& diagonalMatrix, real vScalar, const V3& v)
{
  size_type len = lhs.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) = (vScalar*v(i)) / (coefficientScalar*diagonalMatrix(i));
  }
}

/**************************************************************************************************
	* <summary>
	*  Calculates the sum of two vectors (v1 + v2), uses it as a RHS vector and solve the linear
	*  system formed with a diagonal coefficient matrix, that is, calculates the result of M^(-1) * 
	*  (v1 + v2) (scalars omitted, refer to parameter list for more information). Finally, results 
	*  are stored in the LHS vector.
	* </summary>
	*
	* <param name="lhs">			   	The left hand side vector. </param>
	* <param name="coefficientScalar">	The scalar which multiplies the diagonal coefficient matrix
	* 									before inversion. </param>
	* <param name="diagonalMatrix">   	The diagonal coefficient matrix represented as a vector. </param>
	* <param name="v1Scalar">		   	The scalar which multiplies the first vector parcel. </param>
	* <param name="v1">			   	The first vector parcel. </param>
	* <param name="v2Scalar">		   	The scalar which multiplies the second vector parcel. </param>
	* <param name="v2">			   	The second vector parcel. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3, class V4>
GPU_ONLY void VectorSumAndSolve(V1& lhs, real coefficientScalar, const V2& diagonalMatrix, real v1Scalar, 
                       const V3& v1, real v2Scalar, const V4& v2)
{
  size_type len = lhs.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) = (v1Scalar*v1(i) + v2Scalar*v2(i)) / (coefficientScalar*diagonalMatrix(i));
  }
}

/**************************************************************************************************
	* <summary>
	*  Calculates the sum of two vectors (v1 + v2), uses it as a RHS vector and solve the linear
	*  system formed with a diagonal coefficient matrix, that is, calculates the result of M^(-1) * 
	*  (v1 + v2) (scalars omitted, refer to parameter list for more information). Finally, results 
	*  are accumulated in the LHS vector.
	* </summary>
	*
	* <param name="lhs">			   	The left hand side vector. </param>
	* <param name="coefficientScalar">	The scalar which multiplies the diagonal coefficient matrix
	* 									before inversion. </param>
	* <param name="diagonalMatrix">   	The diagonal coefficient matrix represented as a vector. </param>
	* <param name="v1Scalar">		   	The scalar which multiplies the first vector parcel. </param>
	* <param name="v1">			   	The first vector parcel. </param>
	* <param name="v2Scalar">		   	The scalar which multiplies the second vector parcel. </param>
	* <param name="v2">			   	The second vector parcel. </param>
	**************************************************************************************************/
template<class V1, class V2, class V3, class V4>
GPU_ONLY void VectorAccumulateSumAndSolve(V1& lhs, real coefficientScalar, const V2& diagonalMatrix, real v1Scalar, 
                                 const V3& v1, real v2Scalar, const V4& v2)
{
  size_type len = lhs.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) += ( (v1Scalar*v1(i) + v2Scalar*v2(i)) / (coefficientScalar*diagonalMatrix(i)) );
  }
}

/**********************************************************************************************//**
	* <summary> Solves the linear system formed with a diagonal
	* 			 coefficient matrix and a RHS vector, that is, calculates
	* 			 the result of M^(-1) * v (scalars omitted, refer to
	* 			 parameter list for more information). Finally, results
	* 			 are accumulated in the LHS vector.</summary>
	*
	* <param name="lhs">			    The left hand side vector.</param>
	* <param name="coefficientScalar"> The scalar which multiplies the
	* 									diagonal coefficient matrix
	* 									before inversion.</param>
	* <param name="diagonalMatrix">    The diagonal coefficient matrix
	* 									represented as a vector.</param>
	* <param name="vScalar">		    The scalar which multiplies the
	* 									RHS vector.</param>
	* <param name="v">				    The RHS vector.</param>
	**************************************************************************************************/
template<class V1, class V2, class V3>
GPU_ONLY void VectorAccumulateSolve(V1& lhs, real coefficientScalar, const V2& diagonalMatrix, real vScalar, const V3& v)
{
  size_type len = lhs.Length();
  for (size_type i = 0; i < len; ++i)
  {
    lhs(i) += ( (vScalar*v(i)) / (coefficientScalar*diagonalMatrix(i)) );
  }
}

/**********************************************************************************************//**
	* <summary> Writes to a vector the Voigt notation for a given
	* 			 symmetric second order tensor.</summary>
	*
	* <param name="lhs">			    The vector which will store the
	* 									result.</param>
	* <param name="secondOrderTensor"> The matrix representing the
	* 									second order tensor.</param>
	**************************************************************************************************/
template<class V>
GPU_ONLY void TransformSecondTensorToVoigt(V& lhs, const SymmetricMatrix& secondOrderTensor)
{
  // target vector must have this amount of positions:
  size_type n = secondOrderTensor.Rows();
  // first, write out diagonal elements
  for (size_type i = 0; i < n; ++i)
  {
    lhs(i) = secondOrderTensor(i,i);
  }
  // next, write out off-diagonal elements
  size_type idx = n;
  for (size_type i = n-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = n-1; j > i; --j)	// iterate through columns
    {
      lhs(idx) = secondOrderTensor(i,j);
      ++idx;
    }
  }
}
template<int N, class V>
GPU_ONLY void TransformSecondTensorToVoigt(V& lhs, const AutoSymmetricMatrix<N>& secondOrderTensor)
{
  // target vector must have this amount of positions:
  size_type n = secondOrderTensor.Rows();
  // first, write out diagonal elements
  for (size_type i = 0; i < n; ++i)
  {
    lhs(i) = secondOrderTensor(i,i);
  }
  // next, write out off-diagonal elements
  size_type idx = n;
  for (size_type i = n-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = n-1; j > i; --j)	// iterate through columns
    {
      lhs(idx) = secondOrderTensor(i,j);
      ++idx;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Writes to a vector the Voigt notation for a given
	* 			 symmetric second order tensor.</summary>
	*
	* <param name="lhs">			    The vector which will store the
	* 									result.</param>
	* <param name="secondOrderTensor"> The matrix representing the
	* 									second order tensor.</param>
	* <param name="fullRank">		    The size of an inner tensor in
	* 									which non-zero off-diagonal
	* 									elements are limited to appear.</param>
	**************************************************************************************************/
template<class V>
GPU_ONLY void TransformSecondTensorToVoigt(V& lhs, const SymmetricMatrix& secondOrderTensor, int fullRank)
{
  // target vector must have this amount of positions:
  size_type n = secondOrderTensor.Rows();
  // first, write out diagonal elements
  for (size_type i = 0; i < fullRank; ++i)
  {
    lhs(i) = secondOrderTensor(i,i);
  }
  // next, write out off-diagonal elements
  size_type idx = fullRank;
  for (size_type i = fullRank-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = fullRank-1; j > i; --j)	// iterate through columns
    {
      lhs(idx) = secondOrderTensor(i,j);
      ++idx;
    }
  }
  // then, write out diagonal terms outside the inner tensor
  for (size_type i = fullRank; i < n; ++i)
  {
    lhs(idx) = secondOrderTensor(i,i);
    ++idx;
  }
}
template<int N, class V>
GPU_ONLY void TransformSecondTensorToVoigt(V& lhs, const AutoSymmetricMatrix<N>& secondOrderTensor, int fullRank)
{
  // target vector must have this amount of positions:
  size_type n = secondOrderTensor.Rows();
  // first, write out diagonal elements
  for (size_type i = 0; i < fullRank; ++i)
  {
    lhs(i) = secondOrderTensor(i,i);
  }
  // next, write out off-diagonal elements
  size_type idx = fullRank;
  for (size_type i = fullRank-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = fullRank-1; j > i; --j)	// iterate through columns
    {
      lhs(idx) = secondOrderTensor(i,j);
      ++idx;
    }
  }
  // then, write out diagonal terms outside the inner tensor
  for (size_type i = fullRank; i < n; ++i)
  {
    lhs(idx) = secondOrderTensor(i,i);
    ++idx;
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the product of a matrix and a vector.</summary>
	*
	* <param name="lhs">			  [in,out] The LHS vector which will store the result.</param>
	* <param name="scalar">		  The scalar by which the result will be multiplied before storing in the LHS vector.</param>
	* <param name="matrix">		  The left matrix.</param>
	* <param name="transposeMatrix"> Tells whether the matrix should be transposed.</param>
	* <param name="vector">		  The right vector.</param>
	* <param name="transposeVector"> Tells whether the right vector should be transposed.</param>
	**************************************************************************************************/
template<class V1, class V2, class Matrix>
GPU_ONLY void VectorProduct(V1& lhs, real scalar, const Matrix& matrix, Transposition transposeMatrix, 
                   const V2& vector, Transposition transposeVector)
{
  bool matrixTransposed = (transposeMatrix == Transposed);
  bool vectorTransposed = (transposeVector == Transposed);
  size_type rows = matrixTransposed? matrix.Columns() : matrix.Rows();
  size_type cols = vectorTransposed? vector.Rows() : vector.Columns();
  size_type productLen = matrixTransposed? matrix.Rows() : matrix.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real x_ij = 0;
    for (size_type k = 0; k < productLen; ++k)
    {
      x_ij += (matrixTransposed? matrix(k, i) : matrix(i, k)) * vector(k);				
    }
    lhs(i) = scalar * x_ij;
  }
}

template <class V1, class V2>
GPU_ONLY void VectorProduct(DenseMatrix& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) = l_ij*right(j);
    }
  }
}
template <int M, int N, class V1, class V2>
GPU_ONLY void VectorProduct(AutoDenseMatrix<M,N>& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) = l_ij*right(j);
    }
  }
}

template <class V1, class V2>
GPU_ONLY void VectorProduct(SymmetricMatrix& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) = l_ij*right(j);
    }
  }
}
template <int N, class V1, class V2>
GPU_ONLY void VectorProduct(AutoSymmetricMatrix<N>& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) = l_ij*right(j);
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the product of a matrix and a vector.</summary>
	*
	* <param name="lhs">			  [in,out] The LHS vector which will store the result.</param>
	* <param name="scalar">		  The scalar by which the result will be multiplied before storing in the LHS vector.</param>
	* <param name="matrix">		  The left matrix.</param>
	* <param name="vector">		  The right vector.</param>
	**************************************************************************************************/
template<class V1, class V2, class Matrix>
GPU_ONLY void VectorProduct(V1& lhs, real scalar, const Matrix& matrix, const V2& vector)
{
	size_type rows = matrix.Rows();
	size_type cols = vector.Columns();
	size_type productLen = matrix.Columns();
	// calculate product
	for (size_type i = 0; i < rows; ++i)
	{
		real x_ij = 0;
		for (size_type k = 0; k < productLen; ++k)
		{
			x_ij += matrix(i, k) * vector(k);				
		}
		lhs(i) = scalar * x_ij;
	}
}

/**********************************************************************************************//**
	* <summary> Accumulates the product of a matrix and a vector.</summary>
	*
	* <param name="lhs">			  [in,out] The LHS vector which will store the result.</param>
	* <param name="scalar">		  The scalar by which the result will be multiplied before storing in the LHS vector.</param>
	* <param name="matrix">		  The left matrix.</param>
	* <param name="transposeMatrix"> Tells whether the matrix should be transposed.</param>
	* <param name="vector">		  The right vector.</param>
	* <param name="transposeVector"> Tells whether the right vector should be transposed.</param>
	**************************************************************************************************/
template<class Matrix, class V1, class V2>
GPU_ONLY void VectorAccumulateProduct(V1& lhs, real scalar, const Matrix& matrix, Transposition transposeMatrix, 
                             const V2& vector, Transposition transposeVector)
{
  bool matrixTransposed = (transposeMatrix == Transposed);
  bool vectorTransposed = (transposeVector == Transposed);

  size_type rows = matrixTransposed? matrix.Columns() : matrix.Rows();
  size_type cols = vectorTransposed? vector.Rows() : vector.Columns();
  size_type productLen = matrixTransposed? matrix.Rows() : matrix.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real x_ij = 0;
    for (size_type k = 0; k < productLen; ++k)
    {
      x_ij += (matrixTransposed? matrix(k, i) : matrix(i, k)) * vector(k);				
    }
    lhs(i) += scalar * x_ij;
  }
}

template <class V1, class V2>
GPU_ONLY void VectorAccumulateProduct(DenseMatrix& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) += l_ij*right(j);
    }
  }
}
template <int M, int N, class V1, class V2>
GPU_ONLY void VectorAccumulateProduct(AutoDenseMatrix<M,N>& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) += l_ij*right(j);
    }
  }
}

template <class V1, class V2>
GPU_ONLY void VectorAccumulateProduct(SymmetricMatrix& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) += l_ij*right(j);
    }
  }
}
template <int N, class V1, class V2>
GPU_ONLY void VectorAccumulateProduct(AutoSymmetricMatrix<N>& lhs, real scalar, const V1& left, const V2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    real l_ij = scalar * left(i);
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) += l_ij*right(j);
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the product of a matrix and a vector.</summary>
	*
	* <param name="lhs">			  [in,out] The LHS vector which will store the result.</param>
	* <param name="scalar">		  The scalar by which the result will be multiplied before storing in the LHS vector.</param>
	* <param name="matrix">		  The left matrix.</param>
	* <param name="vector">		  The right vector.</param>
	**************************************************************************************************/
template<class Matrix, class V1, class V2>
GPU_ONLY void VectorAccumulateProduct(V1& lhs, real scalar, const Matrix& matrix, const V2& vector)
{
	size_type rows = matrix.Rows();
	size_type cols = vector.Columns();
	size_type productLen = matrix.Columns();
	// calculate product
	for (size_type i = 0; i < rows; ++i)
	{
		real x_ij = 0;
		for (size_type k = 0; k < productLen; ++k)
		{
			x_ij += matrix(i, k) * vector(k);				
		}
		lhs(i) += scalar * x_ij;
	}
}

} } } } // namespace axis::yuzu::foundation::blas

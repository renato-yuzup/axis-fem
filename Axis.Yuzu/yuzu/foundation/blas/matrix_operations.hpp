#pragma once

#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/blas/blas_basic_types.hpp"
#include "yuzu/foundation/blas/DenseMatrix.hpp"
#include "yuzu/foundation/blas/SymmetricMatrix.hpp"
#include "yuzu/foundation/blas/AutoDenseMatrix.hpp"
#include "yuzu/foundation/blas/AutoSymmetricMatrix.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace blas {

/**********************************************************************************************//**
	* <summary> Calculates the product between two matrices.</summary>
	*
	* <param name="lhs">			 The matrix which will store the
	* 								 result.</param>
	* <param name="scalar">		 The scalar by which the product will be multiplied.</param>
	* <param name="left">			 The first (left) matrix.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								  transpose of the first matrix.</param>
	* <param name="right">			 The second (right) matrix.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								  transpose of the second matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Product(DenseMatrix& lhs, real scalar, const T1& left, Transposition transposeLeft, 
             const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = rightTransposed? right.Rows() : right.Columns();
  size_type productLen = leftTransposed? left.Rows() : left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
          (rightTransposed? right(j, k) : right(k, j));
      }
      lhs(i, j) = scalar * x_ij;
    }
  }
}
template <int M, int N, class T1, class T2> GPU_ONLY 
void Product(AutoDenseMatrix<M, N>& lhs, real scalar, const T1& left, Transposition transposeLeft, 
             const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = rightTransposed? right.Rows() : right.Columns();
  size_type productLen = leftTransposed? left.Rows() : left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
          (rightTransposed? right(j, k) : right(k, j));
      }
      lhs(i, j) = scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the product between two matrices.</summary>
	*
	* <param name="lhs">    The matrix which will store the result.</param>
	* <param name="scalar"> The scalar by which the product will be
	* 						 multiplied.</param>
	* <param name="left">   The first (left) matrix.</param>
	* <param name="right">  The second (right) matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Product(DenseMatrix& lhs, real scalar, const T1& left, const T2& right)
  {
	size_type rows = left.Rows();
	size_type cols = right.Columns();
	size_type productLen = left.Columns();
	// calculate product
	for (size_type i = 0; i < rows; ++i)
	{
		for (size_type j = 0; j < cols; ++j)
		{
			real x_ij = 0;
			for (size_type k = 0; k < productLen; ++k)
			{
				x_ij += left(i, k) * right(k, j);
			}
			lhs(i, j) = scalar * x_ij;
		}
	}
}
template <int M, int N, class T1, class T2> GPU_ONLY 
  void Product(AutoDenseMatrix<M, N>& lhs, real scalar, const T1& left, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  size_type productLen = left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += left(i, k) * right(k, j);
      }
      lhs(i, j) = scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the product of two matrices assuming the
	* 			 result is a symmetric matrix.</summary>
	*
	* <param name="lhs">			 The matrix which will store the
	* 								 result.</param>
	* <param name="scalar">		 The scalar by which the product will
	* 								 be multiplied.</param>
	* <param name="left">			 The first (left) matrix.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								  transpose of the first matrix.</param>
	* <param name="right">			 The second (right) matrix.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								  transpose of the second matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Product(SymmetricMatrix& lhs, real scalar, const T1& left, Transposition transposeLeft, 
             const T2& right, Transposition transposeRight)
{
	bool leftTransposed = (transposeLeft == Transposed);
	bool rightTransposed = (transposeRight == Transposed);
	size_type rows = leftTransposed? left.Columns() : left.Rows();
	size_type cols = rightTransposed? right.Rows() : right.Columns();
	size_type productLen = leftTransposed? left.Rows() : left.Columns();
	// calculate product
	for (size_type i = 0; i < rows; ++i)
	{
		for (size_type j = i; j < cols; ++j)
		{
			real x_ij = 0;
			for (size_type k = 0; k < productLen; ++k)
			{
				x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
						(rightTransposed? right(j, k) : right(k, j));
			}
			lhs(i, j) = scalar * x_ij;
		}
	}
}
template <int N, class T1, class T2> GPU_ONLY 
  void Product(AutoSymmetricMatrix<N>& lhs, real scalar, const T1& left, Transposition transposeLeft, 
  const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = rightTransposed? right.Rows() : right.Columns();
  size_type productLen = leftTransposed? left.Rows() : left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
          (rightTransposed? right(j, k) : right(k, j));
      }
      lhs(i, j) = scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the product of two matrices assuming the
	* 			 result is a symmetric matrix.</summary>
	*
	* <param name="lhs">    The matrix which will store the result.</param>
	* <param name="scalar"> The scalar by which the product will be
	* 						 multiplied.</param>
	* <param name="left">   The first (left) matrix.</param>
	* <param name="right">  The second (right) matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Product(SymmetricMatrix& lhs, real scalar, const T1& left, const T2& right)
{
	size_type rows =left.Rows();
	size_type cols = right.Columns();
	size_type productLen = left.Columns();
	// calculate product
	for (size_type i = 0; i < rows; ++i)
	{
		for (size_type j = i; j < cols; ++j)
		{
			real x_ij = 0;
			for (size_type k = 0; k < productLen; ++k)
			{
				x_ij += left(i, k) * right(k, j);
			}
			lhs(i, j) = scalar * x_ij;
		}
	}
}
template <int N, class T1, class T2> GPU_ONLY 
  void Product(AutoSymmetricMatrix<N>& lhs, real scalar, const T1& left, const T2& right)
{
  size_type rows =left.Rows();
  size_type cols = right.Columns();
  size_type productLen = left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += left(i, k) * right(k, j);
      }
      lhs(i, j) = scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the product between two matrices.</summary>
	*
	* <param name="lhs">			 The matrix which will accumulate the
	* 								 result.</param>
	* <param name="scalar">		 The scalar by which the product will be multiplied.</param>
	* <param name="left">			 The first (left) matrix.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								  transpose of the first matrix.</param>
	* <param name="right">			 The second (right) matrix.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								  transpose of the second matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateProduct(DenseMatrix& lhs, real scalar, const T1& left, Transposition transposeLeft, 
                       const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = rightTransposed? right.Rows() : right.Columns();
  size_type productLen = leftTransposed? left.Rows() : left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
          (rightTransposed? right(j, k) : right(k, j));
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}
template <int M, int N, class T1, class T2> GPU_ONLY 
  void AccumulateProduct(AutoDenseMatrix<M, N>& lhs, real scalar, const T1& left, Transposition transposeLeft, 
  const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = rightTransposed? right.Rows() : right.Columns();
  size_type productLen = leftTransposed? left.Rows() : left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
          (rightTransposed? right(j, k) : right(k, j));
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the product between two matrices.</summary>
	*
	* <param name="lhs">    The matrix which will accumulate the result.</param>
	* <param name="scalar"> The scalar by which the product will be
	* 						 multiplied.</param>
	* <param name="left">   The first (left) matrix.</param>
	* <param name="right">  The second (right) matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateProduct(DenseMatrix& lhs, real scalar, const T1& left, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  size_type productLen = left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += left(i, k) * right(k, j);
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}
template <int M, int N, class T1, class T2> GPU_ONLY 
  void AccumulateProduct(AutoDenseMatrix<M, N>& lhs, real scalar, const T1& left, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  size_type productLen = left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += left(i, k) * right(k, j);
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the product of two matrices assuming the
	* 			 result is a symmetric matrix.</summary>
	*
	* <param name="lhs">			 The matrix which will accumulate the
	* 								 result.</param>
	* <param name="scalar">		 The scalar by which the product will
	* 								 be multiplied.</param>
	* <param name="left">			 The first (left) matrix.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								  transpose of the first matrix.</param>
	* <param name="right">			 The second (right) matrix.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								  transpose of the second matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateProduct(SymmetricMatrix& lhs, real scalar, const T1& left, Transposition transposeLeft, 
                       const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = rightTransposed? right.Rows() : right.Columns();
  size_type productLen = leftTransposed? left.Rows() : left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
          (rightTransposed? right(j, k) : right(k, j));
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}
template <int N, class T1, class T2> GPU_ONLY 
  void AccumulateProduct(AutoSymmetricMatrix<N>& lhs, real scalar, const T1& left, Transposition transposeLeft, 
  const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = rightTransposed? right.Rows() : right.Columns();
  size_type productLen = leftTransposed? left.Rows() : left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += (leftTransposed? left(k, i) : left(i, k)) * 
          (rightTransposed? right(j, k) : right(k, j));
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the product of two matrices assuming the
	* 			 result is a symmetric matrix.</summary>
	*
	* <param name="lhs">    The matrix which will accumulate the result.</param>
	* <param name="scalar"> The scalar by which the product will be
	* 						 multiplied.</param>
	* <param name="left">   The first (left) matrix.</param>
	* <param name="right">  The second (right) matrix.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateProduct(SymmetricMatrix& lhs, real scalar, const T1& left, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  size_type productLen = left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += left(i, k) * right(k, j);
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}
template <int N, class T1, class T2> GPU_ONLY 
  void AccumulateProduct(AutoSymmetricMatrix<N>& lhs, real scalar, const T1& left, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = right.Columns();
  size_type productLen = left.Columns();
  // calculate product
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      real x_ij = 0;
      for (size_type k = 0; k < productLen; ++k)
      {
        x_ij += left(i, k) * right(k, j);
      }
      lhs(i, j) += scalar * x_ij;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the sum of two matrices.</summary>
	*
	* <param name="lhs">			 The matrix where the result will be
	* 								 stored.</param>
	* <param name="leftScalar">	 The scalar which multiplies the first
	* 								 parcel.</param>
	* <param name="left">			 The first matrix parcel.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	* <param name="rightScalar">    The scalar which multiplies the
	* 								 second parcel.</param>
	* <param name="right">			 the second matrix parcel.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Sum(DenseMatrix& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
         real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) = leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
                  rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}
template <int M, int N, class T1, class T2> GPU_ONLY 
  void Sum(AutoDenseMatrix<M, N>& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
  real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) = leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
        rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the sum of two matrices.</summary>
	*
	* <param name="lhs">			 The matrix where the result will be
	* 								 stored.</param>
	* <param name="leftScalar">	 The scalar which multiplies the first
	* 								 parcel.</param>
	* <param name="left">			 The first matrix parcel.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	* <param name="rightScalar">    The scalar which multiplies the
	* 								 second parcel.</param>
	* <param name="right">			 the second matrix parcel.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Sum(DenseMatrix& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) = leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}
template <int M, int N, class T1, class T2> GPU_ONLY 
  void Sum(AutoDenseMatrix<M, N>& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) = leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the sum of two matrices assuming the result is
	* 			 symmetric.</summary>
	*
	* <param name="lhs">			 The matrix where the result will be
	* 								 stored.</param>
	* <param name="leftScalar">	 The scalar which multiplies the first
	* 								 parcel.</param>
	* <param name="left">			 The first matrix parcel.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	* <param name="rightScalar">    The scalar which multiplies the
	* 								 second parcel.</param>
	* <param name="right">			 the second matrix parcel.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Sum(SymmetricMatrix& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
         real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) = leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
                  rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}
template <int N, class T1, class T2> GPU_ONLY 
  void Sum(AutoSymmetricMatrix<N>& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
  real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) = leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
        rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}

/**********************************************************************************************//**
	* <summary> Calculates the sum of two matrices assuming the result is
	* 			 symmetric.</summary>
	*
	* <param name="lhs">		  The matrix where the result will be
	* 							  stored.</param>
	* <param name="leftScalar">  The scalar which multiplies the first
	* 							  parcel.</param>
	* <param name="left">		  The first matrix parcel.</param>
	* <param name="rightScalar"> The scalar which multiplies the second
	* 							  parcel.</param>
	* <param name="right">		  the second matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void Sum(SymmetricMatrix& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) = leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}
template <int N, class T1, class T2> GPU_ONLY 
  void Sum(AutoSymmetricMatrix<N>& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) = leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the sum of two matrices.</summary>
	*
	* <param name="lhs">			 The matrix which will accumulate the
	* 								 result.</param>
	* <param name="leftScalar">	 The scalar which multiplies the first
	* 								 parcel.</param>
	* <param name="left">			 The first matrix parcel.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	* <param name="rightScalar">    The scalar which multiplies the
	* 								 second parcel.</param>
	* <param name="right">			 the second matrix parcel.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateSum(DenseMatrix& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
                   real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) += leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
             rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}
template <int M, int N, class T1, class T2> GPU_ONLY 
  void AccumulateSum(AutoDenseMatrix<M, N>& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
  real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) += leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
        rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the sum of two matrices.</summary>
	*
	* <param name="lhs">			 The matrix where the result will be
	* 								 stored.</param>
	* <param name="leftScalar">	 The scalar which multiplies the first
	* 								 parcel.</param>
	* <param name="left">			 The first matrix parcel.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	* <param name="rightScalar">    The scalar which multiplies the
	* 								 second parcel.</param>
	* <param name="right">			 the second matrix parcel.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateSum(DenseMatrix& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) += leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}
template <int M, int N, class T1, class T2> GPU_ONLY 
  void AccumulateSum(AutoDenseMatrix<M, N>& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(i, j) += leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the sum of two matrices assuming the result is
	* 			 symmetric.</summary>
	*
	* <param name="lhs">			 The matrix where the result will be
	* 								 stored.</param>
	* <param name="leftScalar">	 The scalar which multiplies the first
	* 								 parcel.</param>
	* <param name="left">			 The first matrix parcel.</param>
	* <param name="transposeLeft">  Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	* <param name="rightScalar">    The scalar which multiplies the
	* 								 second parcel.</param>
	* <param name="right">			 the second matrix parcel.</param>
	* <param name="transposeRight"> Tells whether it should be used the
	* 								 transpose of the first matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateSum(SymmetricMatrix& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
                   real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) += leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
             rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}
template <int N, class T1, class T2> GPU_ONLY 
  void AccumulateSum(AutoSymmetricMatrix<N>& lhs, real leftScalar, const T1& left, Transposition transposeLeft, 
  real rightScalar, const T2& right, Transposition transposeRight)
{
  bool leftTransposed = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type rows = leftTransposed? left.Columns() : left.Rows();
  size_type cols = leftTransposed? left.Rows() : left.Columns();
  size_type rightRows = rightTransposed? right.Columns() : right.Rows();
  size_type rightCols = rightTransposed? right.Rows() : right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) += leftScalar * (leftTransposed? left(j,i) : left(i,j)) + 
        rightScalar * (rightTransposed? right(j,i) : right(i,j));
    }
  }
}

/**********************************************************************************************//**
	* <summary> Accumulates the sum of two matrices assuming the result is
	* 			 symmetric.</summary>
	*
	* <param name="lhs">		  The matrix where the result will be
	* 							  stored.</param>
	* <param name="leftScalar">  The scalar which multiplies the first
	* 							  parcel.</param>
	* <param name="left">		  The first matrix parcel.</param>
	* <param name="rightScalar"> The scalar which multiplies the second
	* 							  parcel.</param>
	* <param name="right">		  the second matrix parcel.</param>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
void AccumulateSum(SymmetricMatrix& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) += leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}
template <int N, class T1, class T2> GPU_ONLY 
  void AccumulateSum(AutoSymmetricMatrix<N>& lhs, real leftScalar, const T1& left, real rightScalar, const T2& right)
{
  size_type rows = left.Rows();
  size_type cols = left.Columns();
  size_type rightRows = right.Rows();
  size_type rightCols = right.Columns();
  // calculate sum
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < cols; ++j)
    {
      lhs(i, j) += leftScalar*left(i,j) + rightScalar*right(i,j);
    }
  }
}

/**********************************************************************************************//**
	* <summary> Extracts the symmetric part of a square matrix.</summary>
	*
	* <param name="lhs">		   The matrix which will store the
	* 							   symmetric part.</param>
	* <param name="squareMatrix"> The square matrix to decompose.</param>
	**************************************************************************************************/
template <class Matrix> GPU_ONLY 
void DecomposeSymmetric(SymmetricMatrix& lhs, const Matrix& squareMatrix)
{
  size_t rows = squareMatrix.Rows();
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < rows; ++j)
    {
      lhs(i,j) = 0.5 * (squareMatrix(i,j) + squareMatrix(j,i));
    }
  }
}
template <int N, class Matrix> GPU_ONLY 
  void DecomposeSymmetric(AutoSymmetricMatrix<N>& lhs, const Matrix& squareMatrix)
{
  size_t rows = squareMatrix.Rows();
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = i; j < rows; ++j)
    {
      lhs(i,j) = 0.5 * (squareMatrix(i,j) + squareMatrix(j,i));
    }
  }
}

/**********************************************************************************************//**
	* <summary> Extracts the skew part of a square matrix.</summary>
	*
	* <param name="lhs">		   The matrix which will store the skew
	* 							   part.</param>
	* <param name="squareMatrix"> The square matrix to decompose.</param>
	**************************************************************************************************/
template <class Matrix> GPU_ONLY 
void DecomposeSkew(DenseMatrix& lhs, const Matrix& squareMatrix)
{
  size_t rows = squareMatrix.Rows();
  for (size_type i = 0; i < rows; ++i)
  {
    lhs(i,i) = 0;
    for (size_type j = i+1; j < rows; ++j)
    {
      real x = 0.5 * (squareMatrix(i,j) - squareMatrix(j,i));
      lhs(i,j) = x;
      lhs(j,i) = -x;
    }
  }
}
template <int N, class Matrix> GPU_ONLY 
  void DecomposeSkew(AutoDenseMatrix<N, N>& lhs, const Matrix& squareMatrix)
{
  size_t rows = squareMatrix.Rows();
  for (size_type i = 0; i < rows; ++i)
  {
    lhs(i,i) = 0;
    for (size_type j = i+1; j < rows; ++j)
    {
      real x = 0.5 * (squareMatrix(i,j) - squareMatrix(j,i));
      lhs(i,j) = x;
      lhs(j,i) = -x;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Writes to a symmetric tensor values stored in a vector
	* 			 following Voigt notation.</summary>
	*
	* <param name="lhs">    The symmetric second order tensor which will
	* 						 receive the conversion result.</param>
	* <param name="vector"> The vector which contains the representation
	* 						 in Voigt notation.</param>
	**************************************************************************************************/
template <class Vector> GPU_ONLY 
void TransformVoigtToSecondTensor(SymmetricMatrix& lhs, const Vector& vector)
{
  // target matrix must have this amount of positions:
  size_type vSize = vector.Length();
  size_type n = (size_type)((-1 + sqrt((double)(1 + 8*vSize))) / 2);

  // first, write out diagonal elements
  for (size_type i = 0; i < n; ++i)
  {
    lhs(i,i) = vector(i);
  }

  // next, write out off-diagonal elements
  size_type idx = n;
  for (size_type i = n-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = n-1; j > i; --j)	// iterate through columns
    {
      lhs(i,j) = vector(idx);
      ++idx;
    }
  }
}
template <int N, class Vector> GPU_ONLY 
  void TransformVoigtToSecondTensor(AutoSymmetricMatrix<N>& lhs, const Vector& vector)
{
  // target matrix must have this amount of positions:
  size_type vSize = vector.Length();
  size_type n = (size_type)((-1 + sqrt((double)(1 + 8*vSize))) / 2);

  // first, write out diagonal elements
  for (size_type i = 0; i < n; ++i)
  {
    lhs(i,i) = vector(i);
  }

  // next, write out off-diagonal elements
  size_type idx = n;
  for (size_type i = n-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = n-1; j > i; --j)	// iterate through columns
    {
      lhs(i,j) = vector(idx);
      ++idx;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Writes to a symmetric tensor values stored in a vector
	* 			 following Voigt notation.</summary>
	*
	* <param name="lhs">	   The symmetric second order tensor which
	* 						   will receive the conversion result.</param>
	* <param name="vector">   The vector which contains the
	* 						   representation in Voigt notation.</param>
	* <param name="fullRank"> The size of an inner tensor in which non-
	* 						   zero off-diagonal elements are limited to
	* 						   appear.</param>
	**************************************************************************************************/
template <class Vector> GPU_ONLY 
void TransformVoigtToSecondTensor(SymmetricMatrix& lhs, const Vector& vector, int fullRank)
{
  // target matrix must have this amount of positions:
  size_type vSize = vector.Length();
  size_type n = vSize - (fullRank * (fullRank - 1)) / 2;

  // first, write out diagonal elements
  for (size_type i = 0; i < fullRank; ++i)
  {
    lhs(i,i) = vector(i);
  }

  // next, write out off-diagonal elements
  size_type idx = fullRank;
  for (size_type i = fullRank-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = fullRank-1; j > i; --j)	// iterate through columns
    {
      lhs(i,j) = vector(idx);
      ++idx;
    }
  }

  // then, write out diagonal elements outside inner tensor
  for (size_type i = fullRank; i < n; ++i)
  {
    lhs(i,i) = vector(idx);
    ++idx;
  }

  // lastly, zero out other positions
  for (size_type i = fullRank; i < n; ++i)
  {
    for (size_type j = 0; j < i; ++j)
    {
      lhs(i,j) = 0;
    }
  }
}
template <int N, class Vector> GPU_ONLY 
  void TransformVoigtToSecondTensor(AutoSymmetricMatrix<N>& lhs, const Vector& vector, int fullRank)
{
  // target matrix must have this amount of positions:
  size_type vSize = vector.Length();
  size_type n = vSize - (fullRank * (fullRank - 1)) / 2;

  // first, write out diagonal elements
  for (size_type i = 0; i < fullRank; ++i)
  {
    lhs(i,i) = vector(i);
  }

  // next, write out off-diagonal elements
  size_type idx = fullRank;
  for (size_type i = fullRank-2; i >= 0; --i)	// iterate through rows
  {
    for (size_type j = fullRank-1; j > i; --j)	// iterate through columns
    {
      lhs(i,j) = vector(idx);
      ++idx;
    }
  }

  // then, write out diagonal elements outside inner tensor
  for (size_type i = fullRank; i < n; ++i)
  {
    lhs(i,i) = vector(idx);
    ++idx;
  }

  // lastly, zero out other positions
  for (size_type i = fullRank; i < n; ++i)
  {
    for (size_type j = 0; j < i; ++j)
    {
      lhs(i,j) = 0;
    }
  }
}

/**********************************************************************************************//**
	* <summary> Writes the transpose of a matrix to another one.</summary>
	*
	* <param name="lhs">    [in,out] The matrix which will receive the transpose.</param>
	* <param name="matrix"> The matrix to transpose.</param>
	**************************************************************************************************/
template <class Matrix> GPU_ONLY 
void Transpose(DenseMatrix& lhs, const Matrix& matrix)
{
  size_type rows = matrix.Rows();
  size_type cols = matrix.Columns();
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(j,i) = matrix(i,j);
    }
  }
}
template <int M, int N, class Matrix> GPU_ONLY 
  void Transpose(AutoDenseMatrix<M, N>& lhs, const Matrix& matrix)
{
  size_type rows = matrix.Rows();
  size_type cols = matrix.Columns();
  for (size_type i = 0; i < rows; ++i)
  {
    for (size_type j = 0; j < cols; ++j)
    {
      lhs(j,i) = matrix(i,j);
    }
  }
}

/**************************************************************************************************
	* <summary>
	*  Calculates the double contraction (also known as tensor dot product or inner product) between
	*  two second-order square tensors.
	* </summary>
	*
	* <param name="leftScalar">		The scalar which multiplies the first tensor. </param>
	* <param name="leftTensor">		The first (left) tensor. </param>
	* <param name="transposeLeft"> 	Tells whether it should be used the transpose of the left
	* 									tensor. </param>
	* <param name="rightScalar">   	The scalar which multiplies the right tensor. </param>
	* <param name="rightTensor">   	The second (right) tensor. </param>
	* <param name="transposeRight">	Tells whether it should be used the transpose of the right
	* 									tensor. </param>
	*
	* <returns>A scalar value consisting of the double contraction of the tensor. </returns>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
real DoubleContraction(real leftScalar, const T1& leftTensor, Transposition transposeLeft, 
                       real rightScalar, const T2& rightTensor, Transposition transposeRight)
{
  bool leftTransposed  = (transposeLeft == Transposed);
  bool rightTransposed = (transposeRight == Transposed);
  size_type leftRows  = leftTransposed? leftTensor.Columns() : leftTensor.Rows();
  size_type leftCols  = leftTransposed? leftTensor.Rows() : leftTensor.Columns();
  size_type rightRows = rightTransposed? rightTensor.Columns() : rightTensor.Rows();
  size_type rightCols = rightTransposed? rightTensor.Rows() : rightTensor.Columns();
  real ip = 0;
  for (size_type j = 0; j < leftCols; ++j)
  {
    real x = 0;
    for (size_type i = 0; i < leftRows; ++i)
    {
      x += (leftTransposed? leftTensor(j,i) : leftTensor(i,j)) * 
         (rightTransposed? rightTensor(j,i) : rightTensor(i,j));
    }
    ip += x;
  }

  return ip;
}

/**************************************************************************************************
	* <summary>
	*  Calculates the double contraction (also known as tensor dot product or inner product) between
	*  two second-order square tensors.
	* </summary>
	*
	* <param name="leftScalar"> 	The scalar which multiplies the first tensor. </param>
	* <param name="leftTensor"> 	The first (left) tensor. </param>
	* <param name="rightScalar">	The scalar which multiplies the right tensor. </param>
	* <param name="rightTensor">	The second (right) tensor. </param>
	*
	* <returns>	A scalar value consisting of the double contraction of the tensor. </returns>
	**************************************************************************************************/
template <class T1, class T2> GPU_ONLY 
real DoubleContraction(real leftScalar, const T1& leftTensor, real rightScalar, const T2& rightTensor)
{
  size_type leftRows  = leftTensor.Rows();
  size_type leftCols  = leftTensor.Columns();
  size_type rightRows = rightTensor.Rows();
  size_type rightCols = rightTensor.Columns();
  real ip = 0;
  for (size_type j = 0; j < leftCols; ++j)
  {
    real x = 0;
    for (size_type i = 0; i < leftRows; ++i)
    {
      x += leftTensor(i,j) * rightTensor(i,j);
    }
    ip += x;
  }

  return ip;
}

/**************************************************************************************************
	* <summary>
	*  Calculates the double contraction (also known as tensor dot product or inner product) between
	*  two second-order square tensors.
	* </summary>
	*
	* <param name="lhs"> 	Matrix where result will be written. </param>
	* <param name="m"> 	  The matrix to be inverted. </param>
	**************************************************************************************************/
template <class M1, class M2> GPU_ONLY
void Inverse3D( M1& lhs, const M2& m)
{
  // The coefficients below are organized as follows:
  //      [ a  b  c ]
  //  m = [ d  e  f ]
  //      [ g  h  i ]
  real a = m(0,0); real b = m(0,1); real c = m(0,2); 
  real d = m(1,0); real e = m(1,1); real f = m(1,2); 
  real g = m(2,0); real h = m(2,1); real i = m(2,2);
  real det = a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;

  lhs(0,0) = (e*i - f*h) / det;
  lhs(0,1) = (c*h - b*i) / det;
  lhs(0,2) = (b*f - c*e) / det;
  lhs(1,0) = (f*g - d*i) / det;
  lhs(1,1) = (a*i - c*g) / det;
  lhs(1,2) = (c*d - a*f) / det;
  lhs(2,0) = (d*h - e*g) / det;
  lhs(2,1) = (b*g - a*h) / det;
  lhs(2,2) = (a*e - b*d) / det;
}

/**
 * Calculates the eigenvalues and eigenprojections of a symmetric second-order
 * tensor 3x3.
 *
 * @param [in,out] eigenValues      Array which will contain the distinct 
 *                                  eigenvalues.
 * @param [in,out] eigenProjections Array of pointers to matrices which will
 *                                  store the distinct eigenprojections.
 * @param [in,out] eigenCount       Number of distinct eigenvalues found.
 * @param m                         The matrix representing the second-order
 *                                  tensor from where eigenvalues will be 
 *                                  calculated.
 */
GPU_ONLY void SymmetricEigen(real *eigenValues, 
  DenseMatrix *eigenProjections[], int& eigenCount, const SymmetricMatrix& m);

/**
 * Calculates the tensor natural logarithm of a symmetric second-order tensor
 * 3x3.
 *
 * @param [in,out] lhr The matrix which will store the result.
 * @param m            The second-order tensor logarithm function argument.
 */
GPU_ONLY void SymmetricLogarithm(SymmetricMatrix& lhr, 
  const SymmetricMatrix& m);

/**
 * Calculates the tensor square root of a symmetric second-order tensor
 * 3x3.
 *
 * @param [in,out] lhr The matrix which will store the result.
 * @param m            The second-order tensor square root function argument.
 */
GPU_ONLY void SymmetricSquareRoot(SymmetricMatrix& lhr, 
  const SymmetricMatrix& m);

/**
 * Calculates the tensor exponential of a symmetric second-order tensor
 * 3x3.
 *
 * @param [in,out] lhr The matrix which will store the result.
 * @param m            The second-order tensor exponential function argument.
 */
GPU_ONLY void SymmetricExponential(SymmetricMatrix& lhr, 
  const SymmetricMatrix& m);

} } } } // axis::yuzu::foundation::blas

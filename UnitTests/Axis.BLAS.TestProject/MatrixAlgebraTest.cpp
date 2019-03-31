#include "stdafx.h"

#include "foundation/blas/blas.hpp"
#include "foundation/DimensionMismatchException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/blas/RowVector.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "System.hpp"

namespace afb = axis::foundation::blas;

namespace axis_blas_unit_tests
{
	TEST_CLASS(MatrixAlgebraTest)
	{
	public:
    TEST_METHOD_INITIALIZE(SetUpTest)
    {
      axis::System::Initialize();
    }
    TEST_METHOD_CLEANUP(TearDownTest)
    {
      axis::System::Finalize();
    }
		TEST_METHOD(TestMatrixProduct_ab)
		{
			afb::DenseMatrix m1(2,3);
			afb::DenseMatrix m2(3,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 2; m1(0,1) = 5; m1(0,2) = 9;
			m1(1,0) = 8; m1(1,1) =-1; m1(1,2) = 3;

			m2(0,0) = -3; m2(1,0) = 2; m2(2,0) = 6;
			m2(0,1) =  7; m2(1,1) = 4; m2(2,1) = 1;

			// calculate product using first version
			afb::Product(r, 1.0, m1, afb::NotTransposed, m2, afb::NotTransposed);

			Assert::AreEqual(58, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(43, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-8, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(55, r(1,1), REAL_TOLERANCE);

			// calculate product using alternate version
			r.ClearAll();
			afb::Product(r, 1.0, m1, m2);

			Assert::AreEqual(58, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(43, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-8, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(55, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixProduct_aTb)
		{
			afb::DenseMatrix m1(2,2);
			afb::DenseMatrix m2(2,3);
			afb::DenseMatrix r(2,3);

			m1(0,0) = 3; m1(0,1) = 7;
			m1(1,0) = 4; m1(1,1) = 5;

			m2(0,0) = 4; m2(0,1) = 6; m2(0,2) = 3;
			m2(1,0) = 2; m2(1,1) = 0; m2(1,2) = 1;

			afb::Product(r, 1.0, m1, afb::Transposed, m2, afb::NotTransposed);

			Assert::AreEqual(20, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(18, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(13, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual(38, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(42, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(26, r(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixProduct_abT)
		{
			afb::DenseMatrix m1(2,3);
			afb::DenseMatrix m2(2,3);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 1; m1(0,1) = 9; m1(0,2) = 4;
			m1(1,0) = 9; m1(1,1) = 0; m1(1,2) = 6;

			m2(0,0) = 5; m2(0,1) = 6; m2(0,2) = 5;
			m2(1,0) = 2; m2(1,1) = 9; m2(1,2) = -6;

			afb::Product(r, 1.0, m1, afb::NotTransposed, m2, afb::Transposed);

			Assert::AreEqual( 79, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 59, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 75, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-18, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixProduct_aTbT)
		{
			afb::DenseMatrix m1(2,2);
			afb::DenseMatrix m2(2,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 3; m1(0,1) = 4;
			m1(1,0) = 0; m1(1,1) = 2;

			m2(0,0) = 6; m2(0,1) = 2;
			m2(1,0) = 3; m2(1,1) = 5;

			afb::Product(r, 2.0, m1, afb::Transposed, m2, afb::Transposed);

			Assert::AreEqual(36, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(18, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(56, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(44, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixProduct_ErrorIncompatible)
		{
			afb::DenseMatrix m1(3,2), m2(4,3);
			afb::DenseMatrix r1(3,3);

			// test first version
			try
			{
				// must fail on dimension mismatch
				afb::Product(r1, 1.0, m1, afb::NotTransposed, m2, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] * m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			afb::DenseMatrix m3(2,3), m4(3,3);
			afb::DenseMatrix r2(2,5);
			try
			{
				// must fail on result dimension mismatch
				afb::Product(r2, 1.0, m3, afb::NotTransposed, m4, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] * m2[3x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			// test second version
			try
			{
				// must fail on dimension mismatch
				afb::Product(r1, 1.0, m1, m2);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] * m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on result dimension mismatch
				afb::Product(r2, 1.0, m3, m4);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] * m2[3x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixSymmetricProduct_ab)
		{
			afb::DenseMatrix m1(2,3), m2(3,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 4; m1(0,1) = -1; m1(0,2) = 6;
			m1(1,0) = 3; m1(1,1) =  1; m1(1,2) = 2;

			m2(0,0) = 1.4; m2(1,0) = 0; m2(2,0) = -0.6;
			m2(0,1) = 2.4; m2(1,1) = 0; m2(2,1) = -1.1;

			// calculate using first version
			afb::Product(r, 1.0, m1, afb::NotTransposed, m2, afb::NotTransposed);

			// we don't need to check every position because of the
			// symmetry
			Assert::AreEqual(2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(5, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(3, r(0,1), REAL_TOLERANCE);

			// calculate using alternate version
			r.ClearAll();
			afb::Product(r, 1.0, m1, m2);

			Assert::AreEqual(2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(5, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(3, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricProduct_aTb)
		{
			afb::DenseMatrix m1(3,2), m2(3,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 4; m1(1,0) = -1; m1(2,0) = 6;
			m1(0,1) = 3; m1(1,1) =  1; m1(2,1) = 2;

			m2(0,0) = (real)1.4; m2(1,0) = 0; m2(2,0) = (real)-0.6;
			m2(0,1) = (real)2.4; m2(1,1) = 0; m2(2,1) = (real)-1.1;

			afb::Product(r, 1.0, m1, afb::Transposed, m2, afb::NotTransposed);

			// we don't need to check every position because of the
			// symmetry
			Assert::AreEqual((real)2.0, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual((real)5.0, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual((real)3.0, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricProduct_abT)
		{
			afb::DenseMatrix m1(2,3), m2(2,3);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 4; m1(0,1) = -1; m1(0,2) = 6;
			m1(1,0) = 3; m1(1,1) =  1; m1(1,2) = 2;

			m2(0,0) = 1.4; m2(0,1) = 0; m2(0,2) = -0.6;
			m2(1,0) = 2.4; m2(1,1) = 0; m2(1,2) = -1.1;

			afb::Product(r, 1.0, m1, afb::NotTransposed, m2, afb::Transposed);

			// we don't need to check every position because of the
			// symmetry
			Assert::AreEqual(2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(5, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(3, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricProduct_aTbT)
		{
			afb::DenseMatrix m1(3,2), m2(2,3);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 4; m1(1,0) = -1; m1(2,0) = 6;
			m1(0,1) = 3; m1(1,1) =  1; m1(2,1) = 2;

			m2(0,0) = 1.4; m2(0,1) = 0; m2(0,2) = -0.6;
			m2(1,0) = 2.4; m2(1,1) = 0; m2(1,2) = -1.1;

			afb::Product(r, 1.0, m1, afb::Transposed, m2, afb::Transposed);

			// we don't need to check every position because of the
			// symmetry
			Assert::AreEqual(2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(5, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(3, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricProduct_ErrorIncompatible)
		{
			afb::DenseMatrix m1(3,2), m2(4,3), m3(2,3), m4(3,2), m5(2,3), m6(3,3);
			afb::SymmetricMatrix r1(3), r2(5), r3(2);

			// test version 1
			try
			{
				// must fail on dimension mismatch
				afb::Product(r1, 1.0, m1, afb::NotTransposed, m2, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] * m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on result dimension mismatch
				afb::Product(r2, 1.0, m3, afb::NotTransposed, m4, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] * m2[3x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on a not square result
				afb::Product(r3, 1.0, m5, afb::NotTransposed, m6, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted not square result."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}


			// test version 2
			try
			{
				// must fail on dimension mismatch
				afb::Product(r1, 1.0, m1, m2);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] * m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on result dimension mismatch
				afb::Product(r2, 1.0, m3, m4);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] * m2[3x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on a not square result
				afb::Product(r3, 1.0, m5, m6);
				Assert::Fail(_T("Test failed -- accepted not square result."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}

		TEST_METHOD(TestMatrixVectorVectorProduct)
		{
			afb::ColumnVector v1(3);
			afb::RowVector v2(3);
			afb::DenseMatrix r(3,3);

			// populate vectors
			v1(0) = 3; v1(1) = 6; v1(2) = -1; 
			v2(0) = 5; v2(1) = 4; v2(2) = 7; 

			// calculate using first version...
			afb::VectorProduct(r, 1.0, v1, v2);

			// ...and check
			Assert::AreEqual(15, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(12, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(21, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual(30, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(24, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(42, r(1,2), REAL_TOLERANCE);
			Assert::AreEqual(-5, r(2,0), REAL_TOLERANCE);
			Assert::AreEqual(-4, r(2,1), REAL_TOLERANCE);
			Assert::AreEqual(-7, r(2,2), REAL_TOLERANCE);

			// calculate using alternate version...
			r.ClearAll();
			afb::VectorProduct(r, 1.0, v1, v2);

			// ...and check
			Assert::AreEqual(15, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(12, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(21, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual(30, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(24, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(42, r(1,2), REAL_TOLERANCE);
			Assert::AreEqual(-5, r(2,0), REAL_TOLERANCE);
			Assert::AreEqual(-4, r(2,1), REAL_TOLERANCE);
			Assert::AreEqual(-7, r(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorVectorProduct_ErrorIncompatible)
		{
			// check for dimension incompatibility between product factors
			afb::ColumnVector c1(5), c2(3);
			afb::RowVector    v1(4), v2(3);
			afb::DenseMatrix  r1(5,5), r2(3,3);
			try
			{
				afb::VectorProduct(r1, 1.0, c1, v1);
				Assert::Fail(_T("Incompatible dimension was not caught (v1[5x1] * v2[1x4])."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			// check for incompatibility in the result matrix
			afb::DenseMatrix r3(5,4), r4(3,3);
			afb::ColumnVector c3(4);
			afb::RowVector v3(4);
			try
			{
				afb::VectorProduct(r3, 1.0, c3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x4 but accepted 5x4)."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				afb::VectorProduct(r4, 1.0, c3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x4 but accepted 3x3)."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}

		TEST_METHOD(TestMatrixVectorVectorSymmetricProduct)
		{
			afb::ColumnVector v1(3);
			afb::RowVector v2(3);
			afb::SymmetricMatrix r(3,3);

			// populate vectors
			v1(0) = 4; v1(1) = 8; v1(2) = 6; 
			v2(0) = 2; v2(1) = 4; v2(2) = 3; 

			// calculate...
			afb::VectorProduct(r, 1.0, v1, v2);

			// ...and check
			Assert::AreEqual( 8, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(16, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(12, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual(16, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(32, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(24, r(1,2), REAL_TOLERANCE);
			Assert::AreEqual(12, r(2,0), REAL_TOLERANCE);
			Assert::AreEqual(24, r(2,1), REAL_TOLERANCE);
			Assert::AreEqual(18, r(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorVectorSymmetricProduct_ErrorIncompatible)
		{
			afb::ColumnVector v1(3);
			afb::RowVector v2(3), v3(2);
			afb::SymmetricMatrix r(3,3), r2(4,4);

			try
			{
				// should fail because the result is not a square matrix
				afb::VectorProduct(r, 1.0, v1, v3);
				Assert::Fail(_T("A not square matrix was unexpectedly accepted."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch(...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				afb::VectorProduct(r2, 1.0, v1, v2);
				Assert::Fail(_T("A result matrix of different size than the result was unexpectedly accepted."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch(...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}

		TEST_METHOD(TestMatrixAccumulateProduct_ab)
		{
			afb::DenseMatrix r(2,2), m1(2,2), m2(2,2);

			m1(0,0) = -3; m1(0,1) = 2;
			m1(1,0) =  1; m1(1,1) = 7;

			m2(0,0) = 9; m2(0,1) = 7;
			m2(1,0) =-2; m2(1,1) =-3;

			r(0,0) = 3; r(0,1) =-9;
			r(1,0) =-2; r(1,1) = 7;

			// calculates using first version
			afb::AccumulateProduct(r, 5.0, m1, afb::NotTransposed, m2, afb::NotTransposed);

			Assert::AreEqual(-152, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(-144, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( -27, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( -63, r(1,1), REAL_TOLERANCE);

			// calculate using alternate version
			r(0,0) = 3; r(0,1) =-9;
			r(1,0) =-2; r(1,1) = 7;
			afb::AccumulateProduct(r, 5.0, m1, m2);

			Assert::AreEqual(-152, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(-144, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( -27, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( -63, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateProduct_aTb)
		{
			afb::DenseMatrix r(2,3), m1(3,2), m2(3,3);

			m1(0,0) = -3; m1(1,0) = 8; m1(2,0) = 4;
			m1(0,1) =  6; m1(1,1) = 5; m1(2,1) =-7;

			m2(0,0) = 3; m2(0,1) = 7; m2(0,2) =11;
			m2(1,0) =-2; m2(1,1) = 8; m2(1,2) =-9;
			m2(2,0) =13; m2(2,1) = 5; m2(2,2) =10;

			r(0,0) =-2; r(0,1) = 9; r(0,2) = 7;
			r(1,0) = 8; r(1,1) = 1; r(1,2) = 4; 

			// calculates r = r - 2*m1*m2
			afb::AccumulateProduct(r, -2.0, m1, afb::Transposed, m2, afb::NotTransposed);

			Assert::AreEqual( -56, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(-117, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 137, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual( 174, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( -93, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 102, r(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateProduct_abT)
		{
			afb::DenseMatrix r(2,2), m1(2,3), m2(2,3);

			m1(0,0) = 2; m1(0,1) = 6; m1(0,2) = 7;
			m1(1,0) = 5; m1(1,1) =-9; m1(1,2) = 4; 

			m2(0,0) = 5; m2(0,1) =-4; m2(0,2) =10;
			m2(1,0) = 8; m2(1,1) =-7; m2(1,2) = 1; 

			r(0,0) = 5; r(0,1) = 7;
			r(1,0) = 2; r(1,1) =-4;

			// calculates r = -r + m1*m2
			afb::AccumulateProduct(r, 1.0, m1, afb::NotTransposed, m2, afb::Transposed);

			Assert::AreEqual( 61, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(-12, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(103, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(103, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateProduct_aTbT)
		{
			afb::DenseMatrix r(2,2), m1(2,2), m2(2,2);

			m1(0,0) = 3; m1(0,1) = 5;
			m1(1,0) = 8; m1(1,1) = 7;

			m2(0,0) = 9; m2(0,1) = 1;
			m2(1,0) = 4; m2(1,1) = 0;

			r(0,0) = 5; r(0,1) = 8;
			r(1,0) =10; r(1,1) = 9;

			// calculates r = r + m1*m2
			afb::AccumulateProduct(r, 1.0, m1, afb::Transposed, m2, afb::Transposed);

			Assert::AreEqual(40, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(20, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(62, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(29, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateProduct_ErrorIncompatible)
		{
			afb::DenseMatrix r1(2,2), r2(3,2);
			afb::DenseMatrix m1(2,2), m2(3,2), m3(2,3);

			// test first version
			try
			{
				afb::AccumulateProduct(r1, 1.0, m1, afb::NotTransposed, m2, afb::NotTransposed);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r1[2x2] += m1[2x2]*m2[3x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}

			try
			{
				afb::AccumulateProduct(r2, 1.0, m2, afb::NotTransposed, m3, afb::NotTransposed);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r2[3x2] += m2[3x2]*m3[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}

			// test alternate version
			try
			{
				afb::AccumulateProduct(r1, 1.0, m1, m2);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r1[2x2] += m1[2x2]*m2[3x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}

			try
			{
				afb::AccumulateProduct(r2, 1.0, m2, m3);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r2[3x2] += m2[3x2]*m3[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixSymmetricAccumulateProduct_ab)
		{
			afb::SymmetricMatrix r(2);
			afb::DenseMatrix m1(2,2), m2(2,2);

			m1(0,0) = 2; m1(0,1) = 6;
			m1(1,0) = 4; m1(1,1) =16;

			m2(0,0) =13; m2(0,1) =36;
			m2(1,0) =-1; m2(1,1) =-6;

			r(0,0) = 3; r(0,1) = 4;
			r(1,1) = 8;

			// calculate using first version
			afb::AccumulateProduct(r, 2.0, m1, afb::NotTransposed, m2, afb::NotTransposed);

			Assert::AreEqual( 43, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 76, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(104, r(1,1), REAL_TOLERANCE);

			// calculate using alternate version
			r(0,0) = 3; r(0,1) = 4;
			r(1,1) = 8;
			afb::AccumulateProduct(r, 2.0, m1, m2);

			Assert::AreEqual( 43, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 76, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(104, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateProduct_aTb)
		{
			afb::SymmetricMatrix r(2);
			afb::DenseMatrix m1(2,2), m2(2,2);

			m1(0,0) = 4; m1(0,1) = 6;
			m1(1,0) = 3; m1(1,1) = 5;

			m2(0,0) =-30; m2(0,1) = 54;
			m2(1,0) = 48; m2(1,1) =-52;

			r(0,0) =-6; r(0,1) =12;
			r(1,1) = 2;

			afb::AccumulateProduct(r, 1.0, m1, afb::Transposed, m2, afb::NotTransposed);

			Assert::AreEqual( 18, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 72, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 66, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateProduct_abT)
		{
			afb::SymmetricMatrix r(2);
			afb::DenseMatrix m1(2,2), m2(2,2);

			m1(0,0) = 2; m1(0,1) = 4;
			m1(1,0) = 2; m1(1,1) = 2;

			m2(0,0) =  6; m2(0,1) =  2;
			m2(1,0) = 32; m2(1,1) =-12;

			r(0,0) =-3; r(0,1) =-7;
			r(1,1) =-5;

			afb::AccumulateProduct(r, 3.0, m1, afb::NotTransposed, m2, afb::Transposed);

			Assert::AreEqual( 57, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 41, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(115, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateProduct_aTbT)
		{
			afb::SymmetricMatrix r(2);
			afb::DenseMatrix m1(2,2), m2(2,2);

			m1(0,0) = 3; m1(0,1) = 5;
			m1(1,0) = 2; m1(1,1) = 4;

			m2(0,0) = 70; m2(0,1) =-75;
			m2(1,0) = 16; m2(1,1) =  1;

			r(0,0) =42; r(0,1) =27;
			r(1,1) =49;

			afb::AccumulateProduct(r, 4.0, m1, afb::Transposed, m2, afb::Transposed);

			Assert::AreEqual(282, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(227, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(385, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateProduct_ErrorIncompatible)
		{
			afb::SymmetricMatrix r1(2), r2(4);
			afb::DenseMatrix m1(2,2), m2(3,2), m3(2,3);

			// test first version
			try
			{
				afb::AccumulateProduct(r1, 1.0, m1, afb::NotTransposed, m2, afb::NotTransposed);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r1[2x2] += m1[2x2]*m2[3x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}

			try
			{
				afb::AccumulateProduct(r2, 1.0, m2, afb::NotTransposed, m3, afb::NotTransposed);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r2[4x4] += m2[3x2]*m3[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}

			// test alternate version
			try
			{
				afb::AccumulateProduct(r1, 1.0, m1, m2);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r1[2x2] += m1[2x2]*m2[3x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}

			try
			{
				afb::AccumulateProduct(r2, 1.0, m2,m3);
				Assert::Fail(_T("Accepted matrices of incompatible dimensions: r2[4x4] += m2[3x2]*m3[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}
		}

		TEST_METHOD(TestMatrixVectorVectorAccumulateProduct)
		{
			afb::ColumnVector v1(3);
			afb::RowVector v2(3);
			afb::DenseMatrix r(3,3);

			// populate vectors
			v1(0) = 3; v1(1) = 6; v1(2) = -1; 
			v2(0) = 5; v2(1) = 4; v2(2) = 7; 

			r(0,0) = -1; r(0,1) = 3; r(0,2) = -2;
			r(1,0) = 3; r(1,1) = 6; r(1,2) = 1;
			r(2,0) = 2; r(2,1) = 4; r(2,2) = 5;

			// calculate using first version...
			afb::VectorAccumulateProduct(r, 1.0, v1, v2);

			// ...and check
			Assert::AreEqual(14, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(15, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(19, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual(33, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(30, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(43, r(1,2), REAL_TOLERANCE);
			Assert::AreEqual(-3, r(2,0), REAL_TOLERANCE);
			Assert::AreEqual( 0, r(2,1), REAL_TOLERANCE);
			Assert::AreEqual(-2, r(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorVectorAccumulateProduct_ErrorIncompatible)
		{
			// check for dimension incompatibility between product factors
			afb::ColumnVector c1(5), c2(3);
			afb::RowVector    v1(4), v2(3);
			afb::DenseMatrix  r1(5,5), r2(3,3);
			try
			{
				afb::VectorAccumulateProduct(r1, 1.0, c1, v1);
				Assert::Fail(_T("Incompatible dimension was not caught (v1[5x1] * v2[1x4])."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			// check for incompatibility in the result matrix
			afb::DenseMatrix r3(5,4), r4(3,3);
			afb::ColumnVector c3(4);
			afb::RowVector v3(4);
			try
			{
				afb::VectorAccumulateProduct(r3, 1.0, c3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x4 but accepted 5x4)."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				afb::VectorAccumulateProduct(r4, 1.0, c3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x4 but accepted 3x3)."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixVectorVectorAccumulateSymmetricProduct)
		{
			afb::ColumnVector v1(3);
			afb::RowVector v2(3);
			afb::SymmetricMatrix r(3,3);

			// populate vectors
			v1(0) = 4; v1(1) = 8; v1(2) = 6; 
			v2(0) = 2; v2(1) = 4; v2(2) = 3; 

			r(0,0) = 1; r(1,1) = 2; r(2,2) = -1;
			r(1,0) = 3; r(2,0) = -3; r(2,1) = 4;

			// calculate...
			afb::VectorAccumulateProduct(r, 1.0, v1, v2);

			// ...and check
			Assert::AreEqual( 9, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(19, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual(34, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 9, r(2,0), REAL_TOLERANCE);
			Assert::AreEqual(28, r(2,1), REAL_TOLERANCE);
			Assert::AreEqual(17, r(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorVectorAccumulateSymmetricProduct_ErrorIncompatible)
		{
			afb::ColumnVector v1(3);
			afb::RowVector v2(3), v3(2);
			afb::SymmetricMatrix r(3,3), r2(4,4);

			try
			{
				// should fail because the result is not a square matrix
				afb::VectorAccumulateProduct(r, 1.0, v1, v3);
				Assert::Fail(_T("A not square matrix was unexpectedly accepted."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch(...)
			{
				// this was unexpected...
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixSum_ab)
		{
			afb::DenseMatrix m1(2,3), m2(2,3);
			afb::DenseMatrix r(2,3);

			m1(0,0) = 3; m1(0,1) = 5; m1(0,2) = 7;
			m1(1,0) = 4; m1(1,1) = 1; m1(1,2) = 2;

			m2(0,0) = 2; m2(0,1) = 8; m2(0,2) =  3;
			m2(1,0) = 1; m2(1,1) = 2; m2(1,2) = -1;

			// calculate using first version
			afb::Sum(r, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);

			Assert::AreEqual( 5, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(13, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(10, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual( 5, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 3, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 1, r(1,2), REAL_TOLERANCE);

			// calculate using alternate version
			r.ClearAll();
			afb::Sum(r, 1.0, m1, 1.0, m2);

			Assert::AreEqual( 5, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(13, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(10, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual( 5, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 3, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 1, r(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSum_aTb)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 2; m1(0,1) = 5;
			m1(1,0) = 3; m1(1,1) = 4;

			m2(0,0) = 3; m2(0,1) = 7;
			m2(1,0) = 2; m2(1,1) = 1;

			afb::Sum(r, 1.0, m1, afb::Transposed, 2.0, m2, afb::NotTransposed);

			Assert::AreEqual( 8, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(17, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 9, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 6, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSum_abT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 5; m1(0,1) = 1;
			m1(1,0) = 3; m1(1,1) = 6;

			m2(0,0) = 1; m2(0,1) = 5;
			m2(1,0) = 2; m2(1,1) = 3;

			afb::Sum(r, 1.0, m1, afb::NotTransposed, -3.0, m2, afb::Transposed);

			Assert::AreEqual(  2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( -5, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-12, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( -3, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSum_aTbT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 4; m1(0,1) = 2;
			m1(1,0) = 9; m1(1,1) = 7;

			m2(0,0) = 5; m2(0,1) = 7;
			m2(1,0) = 1; m2(1,1) = 3;

			afb::Sum(r, 1.0, m1, afb::Transposed, -1.0, m2, afb::Transposed);

			Assert::AreEqual(-1, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 8, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-5, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 4, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSum_ErrorIncompatible)
		{
			afb::DenseMatrix m1(3,2), m2(4,3), m3(2,3), m4(2,3);
			afb::DenseMatrix r1(3,2), r2(2,5);

			// test first version
			try
			{
				// must fail on dimension mismatch
				afb::Sum(r1, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] + m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on result dimension mismatch
				afb::Sum(r2, 1.0, m3, afb::NotTransposed, 1.0, m4, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}


			// test alternate version
			try
			{
				// must fail on dimension mismatch
				afb::Sum(r1, 1.0, m1, 1.0, m2);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] + m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on result dimension mismatch
				afb::Sum(r2, 1.0, m3, 1.0, m4);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixSymmetricSum_ab)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 3; m1(0,1) = 7;
			m1(1,0) = 5; m1(1,1) = 1;

			m2(0,0) = -0.5; m2(0,1) = -3;
			m2(1,0) =   -2; m2(1,1) =  4;

			// calculate using first version
			afb::Sum(r, 1.0, m1, afb::NotTransposed, 2.0, m2, afb::NotTransposed);

			Assert::AreEqual(2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(1, r(0,1), REAL_TOLERANCE);

			// calculate using alternate version
			r.ClearAll();
			afb::Sum(r, 1.0, m1, 2.0, m2);

			Assert::AreEqual(2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(1, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricSum_aTb)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 5; m1(0,1) = 8;
			m1(1,0) = 3; m1(1,1) = 4;

			m2(0,0) = 7; m2(0,1) = -2;
			m2(1,0) = 3; m2(1,1) =  1;

			afb::Sum(r, 1.0, m1, afb::Transposed, -1.0, m2, afb::NotTransposed);

			Assert::AreEqual(-2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 3, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 5, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricSum_abT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 1; m1(0,1) = 3;
			m1(1,0) = 2; m1(1,1) = 5;

			m2(0,0) = 3; m2(0,1) = 6;
			m2(1,0) = 5; m2(1,1) = 2;

			afb::Sum(r, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::Transposed);

			Assert::AreEqual(4, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(7, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(8, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricSum_aTbT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 5; m1(0,1) = 2;
			m1(1,0) = 7; m1(1,1) = 3;

			m2(0,0) = -2; m2(0,1) = -3;
			m2(1,0) = -8; m2(1,1) =  1;

			afb::Sum(r, 1.0, m1, afb::Transposed, 1.0, m2, afb::Transposed);

			Assert::AreEqual( 3, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 4, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(-1, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricSum_ErrorIncompatible)
		{
			afb::DenseMatrix m1(2,2), m2(2,3), m3(2,2), m4(2,2);
			afb::SymmetricMatrix r1(2), r2(3);

			// test first version
			try
			{
				// must fail on dimension mismatch
				afb::Sum(r1, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[2x2] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			try
			{
				// must fail on result dimension mismatch
				afb::Sum(r2, 1.0, m3, afb::NotTransposed, 1.0, m4, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[3x3] = m1[2x2] * m2[2x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			// test alternate version
			try
			{
				// must fail on dimension mismatch
				afb::Sum(r1, 1.0, m1, 1.0, m2);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[2x2] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			try
			{
				// must fail on result dimension mismatch
				afb::Sum(r2, 1.0, m3, 1.0, m4);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[3x3] = m1[2x2] * m2[2x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}

		TEST_METHOD(TestMatrixAccumulateSum_ab)
		{
			afb::DenseMatrix m1(2,3), m2(2,3);
			afb::DenseMatrix r(2,3);

			m1(0,0) = 3; m1(0,1) = 5; m1(0,2) = 7;
			m1(1,0) = 4; m1(1,1) = 1; m1(1,2) = 2;

			m2(0,0) = 2; m2(0,1) = 8; m2(0,2) =  3;
			m2(1,0) = 1; m2(1,1) = 2; m2(1,2) = -1;

			r(0,0) = 1; r(0,1) = 3; r(0,2) =  5;
			r(1,0) = -3; r(1,1) = 6; r(1,2) = 7;


			// calculate using first version
			afb::AccumulateSum(r, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);

			Assert::AreEqual( 6, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(16, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(15, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual( 2, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 9, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 8, r(1,2), REAL_TOLERANCE);

			// calculate using alternate version
			r(0,0) = 1; r(0,1) = 3; r(0,2) =  5;
			r(1,0) = -3; r(1,1) = 6; r(1,2) = 7;
			afb::AccumulateSum(r, 1.0, m1, 1.0, m2);

			Assert::AreEqual( 6, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(16, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(15, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual( 2, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 9, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 8, r(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateSum_aTb)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 2; m1(0,1) = 5;
			m1(1,0) = 3; m1(1,1) = 4;

			m2(0,0) = 3; m2(0,1) = 7;
			m2(1,0) = 2; m2(1,1) = 1;

			r(0,0) = 3; r(0,1) = 4;
			r(1,0) = -2; r(1,1) = 1;

			afb::AccumulateSum(r, 1.0, m1, afb::Transposed, 2.0, m2, afb::NotTransposed);

			Assert::AreEqual(11, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(21, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 7, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 7, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateSum_abT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 5; m1(0,1) = 1;
			m1(1,0) = 3; m1(1,1) = 6;

			m2(0,0) = 1; m2(0,1) = 5;
			m2(1,0) = 2; m2(1,1) = 3;

			r(0,0) = 3; r(0,1) = 4;
			r(1,0) = -2; r(1,1) = 1;

			afb::AccumulateSum(r, 1.0, m1, afb::NotTransposed, -3.0, m2, afb::Transposed);

			Assert::AreEqual(  5, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( -1, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-14, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( -2, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateSum_aTbT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::DenseMatrix r(2,2);

			m1(0,0) = 4; m1(0,1) = 2;
			m1(1,0) = 9; m1(1,1) = 7;

			m2(0,0) = 5; m2(0,1) = 7;
			m2(1,0) = 1; m2(1,1) = 3;

			r(0,0) = 3; r(0,1) = 4;
			r(1,0) = -2; r(1,1) = 1;

			afb::AccumulateSum(r, 1.0, m1, afb::Transposed, -1.0, m2, afb::Transposed);

			Assert::AreEqual( 2, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(12, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-7, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 5, r(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixAccumulateSum_ErrorIncompatible)
		{
			afb::DenseMatrix m1(3,2), m2(4,3), m3(2,3), m4(2,3);
			afb::DenseMatrix r1(3,2), r2(2,5);

			// test first version
			try
			{
				// must fail on dimension mismatch
				afb::AccumulateSum(r1, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] + m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on result dimension mismatch
				afb::AccumulateSum(r2, 1.0, m3, afb::NotTransposed, 1.0, m4, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}


			// test alternate version
			try
			{
				// must fail on dimension mismatch
				afb::AccumulateSum(r1, 1.0, m1, 1.0, m2);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[3x2] + m2[4x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				// must fail on result dimension mismatch
				afb::AccumulateSum(r2, 1.0, m3, 1.0, m4);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[2x5] = m1[2x3] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}

		TEST_METHOD(TestMatrixSymmetricAccumulateSum_ab)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 3; m1(0,1) = 7;
			m1(1,0) = 5; m1(1,1) = 1;

			m2(0,0) = -0.5; m2(0,1) = -3;
			m2(1,0) =   -2; m2(1,1) =  4;

			r(0,0) = 3; r(1,1) = -2;
			r(1,0) = 5; 

			// calculate using first version
			afb::AccumulateSum(r, 1.0, m1, afb::NotTransposed, 2.0, m2, afb::NotTransposed);

			Assert::AreEqual(5, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(7, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(6, r(0,1), REAL_TOLERANCE);

			// calculate using alternate version
			r(0,0) = 3; r(1,1) = -2;
			r(1,0) = 5; 
			afb::AccumulateSum(r, 1.0, m1, 2.0, m2);

			Assert::AreEqual(5, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(7, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(6, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateSum_aTb)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 5; m1(0,1) = 8;
			m1(1,0) = 3; m1(1,1) = 4;

			m2(0,0) = 7; m2(0,1) = -2;
			m2(1,0) = 3; m2(1,1) =  1;

			r(0,0) = 3; r(1,1) = -2;
			r(1,0) = 5; 

			afb::AccumulateSum(r, 1.0, m1, afb::Transposed, -1.0, m2, afb::NotTransposed);

			Assert::AreEqual( 1, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 1, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(10, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateSum_abT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 1; m1(0,1) = 3;
			m1(1,0) = 2; m1(1,1) = 5;

			m2(0,0) = 3; m2(0,1) = 6;
			m2(1,0) = 5; m2(1,1) = 2;

			r(0,0) = 3; r(1,1) = -2;
			r(1,0) = 5; 

			afb::AccumulateSum(r, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::Transposed);

			Assert::AreEqual( 7, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 5, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(13, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateSum_aTbT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);
			afb::SymmetricMatrix r(2);

			m1(0,0) = 5; m1(0,1) = 2;
			m1(1,0) = 7; m1(1,1) = 3;

			m2(0,0) = -2; m2(0,1) = -3;
			m2(1,0) = -8; m2(1,1) =  1;

			r(0,0) = 3; r(1,1) = -2;
			r(1,0) = 5; 

			afb::AccumulateSum(r, 1.0, m1, afb::Transposed, 1.0, m2, afb::Transposed);

			Assert::AreEqual(6, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(2, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(4, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricAccumulateSum_ErrorIncompatible)
		{
			afb::DenseMatrix m1(2,2), m2(2,3), m3(2,2), m4(2,2);
			afb::SymmetricMatrix r1(2), r2(3);

			// test first version
			try
			{
				// must fail on dimension mismatch
				afb::AccumulateSum(r1, 1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[2x2] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			try
			{
				// must fail on result dimension mismatch
				afb::AccumulateSum(r2, 1.0, m3, afb::NotTransposed, 1.0, m4, afb::NotTransposed);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[3x3] = m1[2x2] * m2[2x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			// test alternate version
			try
			{
				// must fail on dimension mismatch
				afb::AccumulateSum(r1, 1.0, m1, 1.0, m2);
				Assert::Fail(_T("Test failed -- accepted incompatible dimensions m1[2x2] + m2[2x3]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}

			try
			{
				// must fail on result dimension mismatch
				afb::AccumulateSum(r2, 1.0, m3, 1.0, m4);
				Assert::Fail(_T("Test failed -- accepted incompatible result matrix dimension r[3x3] = m1[2x2] * m2[2x2]."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}

		TEST_METHOD(TestMatrixSymmetricPart)
		{
			afb::DenseMatrix m(3,3);
			afb::SymmetricMatrix r(3);

			m(0,0) = 3; m(0,1) = 2; m(0,2) = 8;
			m(1,0) = 7; m(1,1) = 5; m(1,2) = 3;
			m(2,0) = 9; m(2,1) = 4; m(2,2) = 1;

			afb::DecomposeSymmetric(r, m);

			Assert::AreEqual(        3, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(        5, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(        1, r(2,2), REAL_TOLERANCE);
			Assert::AreEqual((real)3.5, r(1,2), REAL_TOLERANCE);
			Assert::AreEqual((real)8.5, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual((real)4.5, r(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSymmetricPart_ErrorIncompatible)
		{
			afb::DenseMatrix m1(3,3), m2(2,3);
			afb::SymmetricMatrix r(2);

			try
			{
				afb::DecomposeSymmetric(r, m1);
				Assert::Fail(_T("Incompatible matrix was accepted."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				afb::DecomposeSymmetric(r, m2);
				Assert::Fail(_T("Not square matrix was accepted."));
			}
			catch (axis::foundation::ArgumentException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixSkewPart)
		{
			afb::DenseMatrix m(3,3);
			afb::DenseMatrix r(3,3);

			m(0,0) = 7; m(0,1) =  8; m(0,2) = 6;
			m(1,0) = 4; m(1,1) = -1; m(1,2) = 9;
			m(2,0) = 0; m(2,1) =  1; m(2,2) = 2;

			afb::DecomposeSkew(r, m);

			Assert::AreEqual( 0, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 2, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 3, r(0,2), REAL_TOLERANCE);
			Assert::AreEqual(-2, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 0, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 4, r(1,2), REAL_TOLERANCE);
			Assert::AreEqual(-3, r(2,0), REAL_TOLERANCE);
			Assert::AreEqual(-4, r(2,1), REAL_TOLERANCE);
			Assert::AreEqual( 0, r(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixSkewPart_ErrorIncompatible)
		{
			afb::DenseMatrix m1(3,3), m2(4,3);
			afb::DenseMatrix r(4,4);

			try
			{
				afb::DecomposeSkew(r, m1);
				Assert::Fail(_T("Incompatible matrix was accepted."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				afb::DecomposeSkew(r, m2);
				Assert::Fail(_T("Not square matrix was accepted."));
			}
			catch (axis::foundation::ArgumentException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixSymmetricSkewSum)
		{
			afb::DenseMatrix m(3,3);
			afb::DenseMatrix skew(3,3);
			afb::SymmetricMatrix sym(3);
			afb::DenseMatrix sum(3,3);

			m(0,0) = 7; m(0,1) =  8; m(0,2) = 6;
			m(1,0) = 4; m(1,1) = -1; m(1,2) = 9;
			m(2,0) = 0; m(2,1) =  1; m(2,2) = 2;

			// get symmetric and skew parts
			afb::DecomposeSkew(skew, m);
			afb::DecomposeSymmetric(sym, m);

			// sum to get back the original matrix
			afb::Sum(sum, 1.0, sym, 1.0, skew);

			// compare to see if it is correct
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					Assert::AreEqual(m(i,j), sum(i,j));
				}
			}
		}
		
		TEST_METHOD(TestVoigtNotationToSecondTensorFull)
		{
			afb::ColumnVector v(6);
			afb::SymmetricMatrix m(3);

			v(0) = 2; v(1) = 4; v(2) = 7;
			v(3) =-1; v(4) = 0; v(5) = 3;

			afb::TransformVoigtToSecondTensor(m, v);

			Assert::AreEqual( 2, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 4, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 7, m(2,2), REAL_TOLERANCE);
			Assert::AreEqual(-1, m(1,2), REAL_TOLERANCE);
			Assert::AreEqual( 0, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual( 3, m(0,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVoigtNotationToSecondTensorFull_ErrorIncompatible)
		{
			afb::ColumnVector v(6);
			afb::SymmetricMatrix m1(2), m2(4);

			try
			{
				afb::TransformVoigtToSecondTensor(m1, v);
				Assert::Fail(_T("Accepted matrix shorter than required."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				afb::TransformVoigtToSecondTensor(m2, v);
				Assert::Fail(_T("Accepted matrix larger than required."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		TEST_METHOD(TestVoigtNotationToSecondTensorIncomplete)
		{
			afb::ColumnVector v(5);
			afb::SymmetricMatrix m(4);

			v(0) = 5; v(1) = 3; v(2) = 2;
			v(3) = 6; v(4) = -5;

			afb::TransformVoigtToSecondTensor(m, v, 2);

			Assert::AreEqual( 5, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 3, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 2, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 6, m(2,2), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(3,3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVoigtNotationToSecondTensorIncomplete_ErrorIncompatible)
		{
			afb::ColumnVector v(5);
			afb::SymmetricMatrix m1(3), m2(5);

			try
			{
				afb::TransformVoigtToSecondTensor(m1, v, 2);
				Assert::Fail(_T("Accepted matrix shorter than required."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
			try
			{
				afb::TransformVoigtToSecondTensor(m2, v, 2);
				Assert::Fail(_T("Accepted matrix larger than required."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestMatrixTranspose)
		{
			afb::DenseMatrix m(2,3);
			afb::DenseMatrix t(3,2);

			m(0,0) = 4; m(0,1) =-9; m(0,2) = 6;
			m(1,0) =-6; m(1,1) = 7; m(1,2) = 2;

			afb::Transpose(t, m);

			Assert::AreEqual( 4, t(0,0), REAL_TOLERANCE);
			Assert::AreEqual(-9, t(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 6, t(2,0), REAL_TOLERANCE);
			Assert::AreEqual(-6, t(0,1), REAL_TOLERANCE);
			Assert::AreEqual( 7, t(1,1), REAL_TOLERANCE);
			Assert::AreEqual( 2, t(2,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixTranspose_ErrorIncompatible)
		{
			afb::DenseMatrix m(3,4);
			afb::DenseMatrix t(4,5);

			try
			{
				afb::Transpose(t, m);
				Assert::Fail(_T("Accepted matrix with incompatible dimension."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unexpected exception thrown."));
			}
		}
		
		TEST_METHOD(TestSecondTensorInnerProduct_ab)
		{
			afb::DenseMatrix m1(3,3), m2(3,3);

			m1(0,0) = 3; m1(0,1) =-1; m1(0,2) = 7;
			m1(1,0) =-2; m1(1,1) = 8; m1(1,2) = 0;
			m1(2,0) = 6; m1(2,1) = 4; m1(2,2) = 5;

			m2(0,0) = 1; m2(0,1) = 2; m2(0,2) = 3;
			m2(1,0) = 2; m2(1,1) = 9; m2(1,2) = 6;
			m2(2,0) = 7; m2(2,1) = 4; m2(2,2) = 0;

			// calculate using first version
			real y = afb::DoubleContraction(1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);
			Assert::AreEqual(148, y, REAL_TOLERANCE);

			// calculate using alternate version
			real x = afb::DoubleContraction(1.0, m1, 1.0, m2);
			Assert::AreEqual(148, x, REAL_TOLERANCE);
		}
		TEST_METHOD(TestSecondTensorInnerProduct_aTb)
		{
			afb::DenseMatrix m1(3,3), m2(3,3);

			m1(0,0) = 5; m1(0,1) =-3; m1(0,2) = 2;
			m1(1,0) = 6; m1(1,1) = 1; m1(1,2) = 2;
			m1(2,0) = 7; m1(2,1) = 9; m1(2,2) = 4;

			m2(0,0) = 6; m2(0,1) = 2; m2(0,2) =-1;
			m2(1,0) = 3; m2(1,1) = 5; m2(1,2) = 4;
			m2(2,0) = 1; m2(2,1) = 7; m2(2,2) = 0;

			real x = afb::DoubleContraction(1.0, m1, afb::Transposed, 1.0, m2, afb::NotTransposed);
			Assert::AreEqual(83, x, REAL_TOLERANCE);
		}
		TEST_METHOD(TestSecondTensorInnerProduct_abT)
		{
			afb::DenseMatrix m1(2,3), m2(3,2);

			m1(0,0) = 4; m1(0,1) = 2; m1(0,2) = 6;
			m1(1,0) = 3; m1(1,1) = 5; m1(1,2) = 7;

			m2(0,0) = 1; m2(1,0) = 7; m2(2,0) = 9;
			m2(0,1) = 3; m2(1,1) = 2; m2(2,1) =-3;

			real x = afb::DoubleContraction(1.0, m1, afb::NotTransposed, 1.0, m2, afb::Transposed);
			Assert::AreEqual(70, x, REAL_TOLERANCE);
		}
		TEST_METHOD(TestSecondTensorInnerProduct_aTbT)
		{
			afb::DenseMatrix m1(2,2), m2(2,2);

			m1(0,0) = 4; m1(0,1) = 6;
			m1(1,0) = 2; m1(1,1) = 9;

			m2(0,0) =-5; m2(0,1) = 3;
			m2(1,0) = 4; m2(1,1) = 8;

			real x = afb::DoubleContraction(1.0, m1, afb::Transposed, 1.0, m2, afb::Transposed);
			Assert::AreEqual(78, x, REAL_TOLERANCE);
		}
		TEST_METHOD(TestSecondTensorInnerProduct_ErrorIncompatible)
		{
			afb::DenseMatrix m1(2,3), m2(2,4);
			real x;

			// test first version
			try
			{
				x = afb::DoubleContraction(1.0, m1, afb::NotTransposed, 1.0, m2, afb::NotTransposed);
				Assert::Fail(_T("Accepted incompatible matrices."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}

			// test alternate version
			try
			{
				x = afb::DoubleContraction(1.0, m1, 1.0, m2);
				Assert::Fail(_T("Accepted incompatible matrices."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}
		}

    TEST_METHOD(TestSymmetricEigen3D_TrivialCase)
    {
      // Calculate the eigenvalue of a diagonal tensor
      afb::SymmetricMatrix m(3,3);

      m.ClearAll();
      m(0,0) = 1.58;  m(1,1) = 0.474; m(2,2) = 0.800;

      afb::DenseMatrix e0(3,3), e1(3,3), e2(3,3);
      afb::DenseMatrix *ei[] = {&e0, &e1, &e2};
      real x[3];
      int eigenCount = 0;
      afb::SymmetricEigen(x, ei, eigenCount, m);
      Assert::AreEqual(3, eigenCount);

      // Eigenprojection = V*V^T where V is an eigenvector, if all eigenvalues
      // are distinct.
      Assert::AreEqual((real)0.474, x[0], REAL_TOLERANCE);
      Assert::AreEqual((real)1.580, x[1], REAL_TOLERANCE);
      Assert::AreEqual((real)0.800, x[2], REAL_TOLERANCE);

      Assert::AreEqual((real)0.0, e0(0,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e0(0,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e0(0,2), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e0(1,0), REAL_TOLERANCE);
      Assert::AreEqual((real)1.0, e0(1,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e0(1,2), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e0(2,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e0(2,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e0(2,2), REAL_TOLERANCE);

      Assert::AreEqual((real)1.0, e1(0,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(0,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(0,2), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(1,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(1,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(1,2), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(2,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(2,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e1(2,2), REAL_TOLERANCE);

      Assert::AreEqual((real)0.0, e2(0,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e2(0,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e2(0,2), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e2(1,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e2(1,1), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e2(1,2), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e2(2,0), REAL_TOLERANCE);
      Assert::AreEqual((real)0.0, e2(2,1), REAL_TOLERANCE);
      Assert::AreEqual((real)1.0, e2(2,2), REAL_TOLERANCE);
    }

    TEST_METHOD(TestSymmetricEigen3D_FullTensor)
    {
      // Calculate the eigenvalue of a symmetric, full tensor
      afb::SymmetricMatrix m(3,3);
      m(0,0) = 1.58;  m(0,1) = -0.50; m(0,2) = 0.100;
                      m(1,1) = 0.474; m(1,2) = 0.450;
                                      m(2,2) = 0.800;

      afb::DenseMatrix e0(3,3), e1(3,3), e2(3,3);
      afb::DenseMatrix *ei[] = {&e0, &e1, &e2};
      real x[3];
      int eigenCount = 0;
      afb::SymmetricEigen(x, ei, eigenCount, m);
      Assert::AreEqual(3, eigenCount);
      Assert::AreEqual((real)0.015618835562090, x[0], REAL_TOLERANCE*5);
      Assert::AreEqual((real)1.778097418778062, x[1], REAL_TOLERANCE*5);
      Assert::AreEqual((real)1.060283745659849, x[2], REAL_TOLERANCE*5);

      // Eigenprojection = V*V^T where V is an eigenvector, if all eigenvalues
      // are distinct.
      Assert::AreEqual((real) 0.085295237643160, e0(0,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.237448779957277, e0(0,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.147098732065772, e0(0,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.237448779957277, e0(1,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.661020763422663, e0(1,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.409500171725951, e0(1,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.147098732065772, e0(2,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.409500171725951, e0(2,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.253683998934177, e0(2,2), REAL_TOLERANCE*5);

      Assert::AreEqual((real) 0.848160489141514, e1(0,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.350990237534270, e1(0,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.074767151586629, e1(0,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.350990237534270, e1(1,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.145248627378360, e1(1,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.030940536173424, e1(1,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-0.074767151586629, e1(2,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.030940536173424, e1(2,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.006590883480126, e1(2,2), REAL_TOLERANCE*5);

      Assert::AreEqual((real) 0.066544273215325, e2(0,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.113541457576993, e2(0,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.221865883652402, e2(0,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.113541457576993, e2(1,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.193730609198977, e2(1,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.378559635552527, e2(1,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.221865883652402, e2(2,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.378559635552527, e2(2,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.739725117585698, e2(2,2), REAL_TOLERANCE*5);
    }

    TEST_METHOD(TestSymmetricLogarithm)
    {
      afb::SymmetricMatrix m(3,3);
      m(0,0) = 1.58;  m(0,1) = -0.50; m(0,2) = 0.100;
                      m(1,1) = 0.474; m(1,2) = 0.450;
                                      m(2,2) = 0.800;
      afb::SymmetricMatrix logm(3,3);
      afb::SymmetricLogarithm(logm, m);
      Assert::AreEqual((real) 0.137282312529862, logm(0,0), REAL_TOLERANCE*50);
      Assert::AreEqual((real)-1.182979385451586, logm(0,1), REAL_TOLERANCE*50);
      Assert::AreEqual((real) 0.581779958706627, logm(0,2), REAL_TOLERANCE*50);
      Assert::AreEqual((real)-2.654431622587698, logm(1,1), REAL_TOLERANCE*50);
      Assert::AreEqual((real) 1.743192141686580, logm(1,2), REAL_TOLERANCE*50);
      Assert::AreEqual((real)-1.008047891481076, logm(2,2), REAL_TOLERANCE*50);
    }

    TEST_METHOD(TestSymmetricSquareRoot)
    {
      afb::SymmetricMatrix m(3,3);
      m(0,0) = 1.58;  m(0,1) = -0.50; m(0,2) = 0.100;
      m(1,1) = 0.474; m(1,2) = 0.450;
      m(2,2) = 0.800;
      afb::SymmetricMatrix sqrtm(3,3);
      afb::SymmetricSquareRoot(sqrtm, m);
      Assert::AreEqual((real) 1.210162805631847, sqrtm(0,0), REAL_TOLERANCE);
      Assert::AreEqual((real)-0.321440080951622, sqrtm(0,1), REAL_TOLERANCE);
      Assert::AreEqual((real) 0.110373267701337, sqrtm(0,2), REAL_TOLERANCE);
      Assert::AreEqual((real) 0.475778104579598, sqrtm(1,1), REAL_TOLERANCE);
      Assert::AreEqual((real) 0.379883494719710, sqrtm(1,2), REAL_TOLERANCE);
      Assert::AreEqual((real) 0.802188426877669, sqrtm(2,2), REAL_TOLERANCE);
    }

    TEST_METHOD(TestSymmetricExponential)
    {
      afb::SymmetricMatrix m(3,3);
      m(0,0) = 1.58;  m(0,1) = -0.50; m(0,2) = 0.100;
      m(1,1) = 0.474; m(1,2) = 0.450;
      m(2,2) = 0.800;
      afb::SymmetricMatrix expm(3,3);
      afb::SymmetricExponential(expm, m);
      Assert::AreEqual((real) 5.298673917610143, expm(0,0), REAL_TOLERANCE*5);
      Assert::AreEqual((real)-1.508363254208931, expm(0,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.048638953834965, expm(0,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 2.090429647248483, expm(1,1), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 0.860151531999821, expm(1,2), REAL_TOLERANCE*5);
      Assert::AreEqual((real) 2.432413093756330, expm(2,2), REAL_TOLERANCE*5);
    }
	};
}


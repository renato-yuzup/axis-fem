#include "stdafx.h"
#include "foundation/blas/blas.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/blas/RowVector.hpp"
#include "foundation/DimensionMismatchException.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "System.hpp"

namespace afb = axis::foundation::blas;

namespace axis_blas_unit_tests
{
	TEST_CLASS(VectorAlgebraTest)
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
		TEST_METHOD(TestScalarProduct_vv)
		{
			afb::RowVector r(3);
			afb::ColumnVector c(3);

			// fill vectors
			r(0) = 4; r(1) = -2; r(2) = 3;
			c(0) = 1; c(1) = 5; c(2) = 6;

			real sprod = afb::VectorScalarProduct(r,c);
			Assert::AreEqual((real)12.0, sprod, REAL_TOLERANCE);
		}
		TEST_METHOD(TestScalarProduct_ErrorIncompatible)
		{
			afb::RowVector r(3);
			afb::RowVector c(4);

			// fill vectors
			r(0) = 4; r(1) = -2; r(2) = 3;
			c(0) = 1; c(1) = 5; c(2) = 6; c(3) = -4;

			try
			{
				real sprod = afb::VectorScalarProduct(r,c);

				// huh, no error...?
				Assert::Fail(_T("DimensionMismatchException not thrown."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// ok, it was expected, ignore it
			}
			catch(...)
			{
				Assert::Fail(_T("Unexpected exception was thrown."));
			}
		}
		TEST_METHOD(TestVectorScale)
		{
			afb::RowVector r(4);
			afb::RowVector result(4);

			// fill vectors
			r(0) = 3; r(1) = -2; r(2) = 0; r(3) = 7;

			afb::VectorScale(result, 1.5, r);

			Assert::AreEqual((real) 4.5, result(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-3.0, result(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 0.0, result(2), REAL_TOLERANCE);
			Assert::AreEqual((real)10.5, result(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorScale_ErrorIncompatible)
		{
			afb::RowVector r(3);
			afb::RowVector c(4);

			// fill vectors
			c(0) = 1; c(1) = 5; c(2) = 6; c(3) = -4;

			try
			{
				afb::VectorScale(r, 4.0, c);

				// huh, no error...?
				Assert::Fail(_T("DimensionMismatchException not thrown."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// ok, it was expected, ignore it
			}
			catch(...)
			{
				Assert::Fail(_T("Unexpected exception was thrown."));
			}
		}
		TEST_METHOD(TestVectorAssign)
		{
			afb::RowVector r(4);
			afb::RowVector result(4);

			// fill vectors
			r(0) = 3; r(1) = -2; r(2) = 0; r(3) = 7;

			result = r;

			Assert::AreEqual((real) 3.0, result(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-2.0, result(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 0.0, result(2), REAL_TOLERANCE);
			Assert::AreEqual((real) 7.0, result(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorAssign_ErrorIncompatible)
		{
			afb::RowVector r(3);
			afb::RowVector c(4);

			// fill vectors
			c(0) = 1; c(1) = 5; c(2) = 6; c(3) = -4;

			try
			{
				r = c;

				// huh, no error...?
				Assert::Fail(_T("DimensionMismatchException not thrown."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// ok, it was expected, ignore it
			}
			catch(...)
			{
				Assert::Fail(_T("Unexpected exception was thrown."));
			}
		}
		TEST_METHOD(TestVectorSwap)
		{
			afb::RowVector a(4);
			afb::RowVector b(4);

			// fill vectors
			a(0) = 3; a(1) = -2; a(2) = 0; a(3) = 7;
			b(0) = 1.5; b(1) = -6.0; b(2) = -9; b(3) = 8;

			afb::VectorSwap(a, b);

			Assert::AreEqual((real) 1.5, a(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-6.0, a(1), REAL_TOLERANCE);
			Assert::AreEqual((real)-9.0, a(2), REAL_TOLERANCE);
			Assert::AreEqual((real) 8.0, a(3), REAL_TOLERANCE);

			Assert::AreEqual((real) 3.0, b(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-2.0, b(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 0.0, b(2), REAL_TOLERANCE);
			Assert::AreEqual((real) 7.0, b(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorSwap_ErrorIncompatible)
		{
			afb::RowVector r(3);
			afb::RowVector c(4);

			try
			{
				// try swapping
				afb::VectorSwap(r, c);

				// huh, no error...?
				Assert::Fail(_T("DimensionMismatchException not thrown."));
			}
			catch (axis::foundation::DimensionMismatchException&)
			{
				// ok, it was expected, ignore it
			}
			catch(...)
			{
				Assert::Fail(_T("Unexpected exception was thrown."));
			}
		}
		TEST_METHOD(TestVectorSum_1)
		{
			afb::ColumnVector v1(3), v2(3);
			afb::ColumnVector r(3);

			v1(0) = 3; v1(1) = 2; v1(2) = 5;
			v2(0) = 7; v2(1) = 1; v2(2) = 4;

			afb::VectorSum(r, v1, 2.0, v2);

			Assert::AreEqual(17, r(0), REAL_TOLERANCE);
			Assert::AreEqual( 4, r(1), REAL_TOLERANCE);
			Assert::AreEqual(13, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorSum_2)
		{
			afb::ColumnVector v1(3), v2(3);
			afb::ColumnVector r(3);

			v1(0) = 3; v1(1) = 2; v1(2) = 5;
			v2(0) = 7; v2(1) = 1; v2(2) = 4;

			afb::VectorSum(r, -2.0, v1, 1.0, v2);

			Assert::AreEqual( 1, r(0), REAL_TOLERANCE);
			Assert::AreEqual(-3, r(1), REAL_TOLERANCE);
			Assert::AreEqual(-6, r(2), REAL_TOLERANCE);
		}

		TEST_METHOD(TestVectorSum_1_ErrorIncompatible)
		{
			afb::ColumnVector v1(3), v2(4), v4(3);
			afb::ColumnVector r1(3), r2(4);

			// method version 1
			try
			{
				afb::VectorSum(r1, v1, 1.0, v2);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorSum(r2, v1, 1.0, v4);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v4[3x1]"));
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
			
		TEST_METHOD(TestVectorSum_2_ErrorIncompatible)
		{
			afb::ColumnVector v1(3), v2(4), v4(3);
			afb::ColumnVector r1(3), r2(4);

			// method version 2
			try
			{
				afb::VectorSum(r1, 1.0, v1, 1.0, v2);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorSum(r2, -4.0, v1, 1.0, v4);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v4[3x1]"));
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

		TEST_METHOD(TestVectorAccumulateSum_1)
		{
			afb::ColumnVector v1(3), v2(3);
			afb::ColumnVector r(3);

			v1(0) = 3; v1(1) = 2; v1(2) = 5;
			v2(0) = 7; v2(1) = 1; v2(2) = 4;
			r(0) = -1; r(1) = 4; r(2) = 2;

			afb::VectorAccumulateSum(r, v1, 2.0, v2);

			Assert::AreEqual(16, r(0), REAL_TOLERANCE);
			Assert::AreEqual( 8, r(1), REAL_TOLERANCE);
			Assert::AreEqual(15, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorAccumulateSum_1_ErrorIncompatible)
		{
			afb::ColumnVector v1(3), v2(4), v4(3);
			afb::ColumnVector r1(3), r2(4);

			// method version 1
			try
			{
				afb::VectorAccumulateSum(r1, v1, 1.0, v2);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorAccumulateSum(r2, v1, 1.0, v4);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v4[3x1]"));
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
		TEST_METHOD(TestVectorAccumulateSum_2)
		{
			afb::ColumnVector v1(3), v2(3);
			afb::ColumnVector r(3);

			v1(0) = 3; v1(1) = 2; v1(2) = 5;
			v2(0) = 7; v2(1) = 1; v2(2) = 4;
			r(0) = -1; r(1) = 4; r(2) = 2;

			afb::VectorAccumulateSum(r, -1.0, v1, 2.0, v2);

			Assert::AreEqual(10, r(0), REAL_TOLERANCE);
			Assert::AreEqual( 4, r(1), REAL_TOLERANCE);
			Assert::AreEqual( 5, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorAccumulateSum_2_ErrorIncompatible)
		{
			afb::ColumnVector v1(3), v2(4), v4(3);
			afb::ColumnVector r1(3), r2(4);

			// method version 2
			try
			{
				afb::VectorAccumulateSum(r1, 1.0, v1, 1.0, v2);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorAccumulateSum(r2, -4.0, v1, 1.0, v4);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v4[3x1]"));
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
		TEST_METHOD(TestVectorElementProduct)
		{
			afb::ColumnVector a(4), b(4), r(4);

			a(0) = 8; a(1) = 1; a(2) = -3; a(3) = 3.2;
			b(0) = -1; b(1) = 2; b(2) = 4; b(3) = 5;

			afb::VectorElementProduct(r, -1.0, a, b);

			Assert::AreEqual(  8, r(0), REAL_TOLERANCE);
			Assert::AreEqual( -2, r(1), REAL_TOLERANCE);
			Assert::AreEqual( 12, r(2), REAL_TOLERANCE);
			Assert::AreEqual(-16, r(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorElementProduct_ErrorIncompatible)
		{
			afb::ColumnVector a(3), b(4), c(3), r1(3), r2(4);

			try
			{
				afb::VectorElementProduct(r1, -1.0, a, b);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorElementProduct(r2, -4.0, a, c);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v2[3x1]"));
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
		TEST_METHOD(TestVectorElementDivide)
		{
			afb::ColumnVector a(4), b(4), r(4);

			a(0) = 8; a(1) = 1; a(2) = -3; a(3) = 3.2;
			b(0) = -1; b(1) = 2; b(2) = 4; b(3) = 5;

			afb::VectorElementDivide(r, -1.0, a, b);

			Assert::AreEqual((real)  8.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real) -0.5, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 0.75, r(2), REAL_TOLERANCE);
			Assert::AreEqual((real)-0.64, r(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorElementDivide_ErrorIncompatible)
		{
			afb::ColumnVector a(3), b(4), c(3), r1(3), r2(4);

			try
			{
				afb::VectorElementDivide(r1, -1.0, a, b);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorElementDivide(r2, -4.0, a, c);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v2[3x1]"));
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
		TEST_METHOD(TestVectorAccumulateElementProduct)
		{
			afb::ColumnVector a(4), b(4), r(4);

			a(0) = 8; a(1) = 1; a(2) = -3; a(3) = 3.2;
			b(0) = -1; b(1) = 2; b(2) = 4; b(3) = 5;
			r(0) = 3; r(1) = 0.5; r(2) = -4; r(3) = -2;

			afb::VectorAccumulateElementProduct(r, -1.0, a, b);

			Assert::AreEqual((real)  11, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-1.5, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)   8, r(2), REAL_TOLERANCE);
			Assert::AreEqual((real) -18, r(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorAccumulateElementProduct_ErrorIncompatible)
		{
			afb::ColumnVector a(3), b(4), c(3), r1(3), r2(4);

			try
			{
				afb::VectorAccumulateElementProduct(r1, -1.0, a, b);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorAccumulateElementProduct(r2, -4.0, a, c);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v2[3x1]"));
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
		TEST_METHOD(TestVectorAccumulateElementDivide)
		{
			afb::ColumnVector a(4), b(4), r(4);

			a(0) = 8; a(1) = 1; a(2) = -3; a(3) = 3.2;
			b(0) = -1; b(1) = 2; b(2) = 4; b(3) = 5;
			r(0) = 3; r(1) = 0.5; r(2) = -4; r(3) = -2;

			afb::VectorAccumulateElementDivide(r, -1.0, a, b);

			Assert::AreEqual(         11, r(0), REAL_TOLERANCE);
			Assert::AreEqual(          0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)-3.25, r(2), REAL_TOLERANCE);
			Assert::AreEqual((real)-2.64, r(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorAccumulateElementDivide_ErrorIncompatible)
		{
			afb::ColumnVector a(3), b(4), c(3), r1(3), r2(4);

			try
			{
				afb::VectorAccumulateElementDivide(r1, -1.0, a, b);
				Assert::Fail(_T("Accepted incompatible vectors: v1[3x1] + alpha*v2[4x1]"));
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
				afb::VectorAccumulateElementDivide(r2, -4.0, a, c);
				Assert::Fail(_T("Accepted incompatible vectors: r2[4x1] = v1[3x1] + alpha*v2[3x1]"));
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
		TEST_METHOD(TestVectorSolve)
		{
			afb::ColumnVector m(4), b(4), r(4);

			m(0) = 2; m(1) = 3; m(2) = -8; m(3) = 3.2;
			b(0) = 4; b(1) = 6; b(2) = 4; b(3) = 64;
			r(0) = 3; r(1) = 0.5; r(2) = -4; r(3) = -2;

			afb::VectorSolve(r, -1.0, m, 1.0, b);

			Assert::AreEqual(       -2, r(0), REAL_TOLERANCE);
			Assert::AreEqual(       -2, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)0.5, r(2), REAL_TOLERANCE);
			Assert::AreEqual(      -20, r(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorSolve_ErrorIncompatible)
		{
			afb::ColumnVector a(3), b(4), c(3), r1(3), r2(4);

			try
			{
				afb::VectorSolve(r2, -1.0, a, 1.0, b);
				Assert::Fail(_T("Accepted incompatible vectors: r[4] = alpha*a^(-1)[3x1] * beta*v2[4x1]"));
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
				afb::VectorSolve(r2, -1.0, r2, 1.0, c);
				Assert::Fail(_T("Accepted incompatible vectors: r[4] = alpha*a^(-1)[4x1] * beta*v2[3x1]"));
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
				afb::VectorSolve(r1, -1.0, r2, 1.0, a);
				Assert::Fail(_T("Accepted incompatible vectors: r[4] = alpha*a^(-1)[4x1] * beta*v2[3x1]"));
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
		TEST_METHOD(TestVectorAccumulateSolve)
		{
			afb::ColumnVector m(4), b(4), r(4);

			m(0) = 2; m(1) = 3; m(2) = -8; m(3) = 3.2;
			b(0) = 4; b(1) = 6; b(2) = 4; b(3) = 64;
			r(0) = 3; r(1) = 0.5; r(2) = -4; r(3) = -2;

			afb::VectorAccumulateSolve(r, -1.0, m, 1.0, b);

			Assert::AreEqual((real)   1, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-1.5, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)-3.5, r(2), REAL_TOLERANCE);
			Assert::AreEqual((real) -22, r(3), REAL_TOLERANCE);
		}
		TEST_METHOD(TestVectorAccumulateSolve_ErrorIncompatible)
		{
			afb::ColumnVector a(3), b(4), c(3), r1(3), r2(4);

			try
			{
				afb::VectorAccumulateSolve(r2, -1.0, a, 1.0, b);
				Assert::Fail(_T("Accepted incompatible vectors: r[4] = alpha*a^(-1)[3x1] * beta*v2[4x1]"));
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
				afb::VectorAccumulateSolve(r2, -1.0, r2, 1.0, c);
				Assert::Fail(_T("Accepted incompatible vectors: r[4] = alpha*a^(-1)[4x1] * beta*v2[3x1]"));
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
				afb::VectorAccumulateSolve(r1, -1.0, r2, 1.0, a);
				Assert::Fail(_T("Accepted incompatible vectors: r[4] = alpha*a^(-1)[4x1] * beta*v2[3x1]"));
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
		TEST_METHOD(TestVectorTransformTensorToVoigt)
		{
			{
				afb::ColumnVector c(6);
				afb::SymmetricMatrix m(3);
				m(0,0) = 5; m(1,1) = 2; m(2,2) = 3;
				m(1,0) = -4; m(2,0) = 6; m(2,1) = 8;

				// full transformation
				afb::TransformSecondTensorToVoigt(c, m);

				Assert::AreEqual( 5, c(0), REAL_TOLERANCE);
				Assert::AreEqual( 2, c(1), REAL_TOLERANCE);
				Assert::AreEqual( 3, c(2), REAL_TOLERANCE);
				Assert::AreEqual( 8, c(3), REAL_TOLERANCE);
				Assert::AreEqual( 6, c(4), REAL_TOLERANCE);
				Assert::AreEqual(-4, c(5), REAL_TOLERANCE);
			}

			{
				afb::ColumnVector c(7);
				afb::SymmetricMatrix m(4);
				m(0,0) = 5; m(1,1) = 2; m(2,2) = 3; m(3,3) = 7;
				m(1,0) = -4; m(2,0) = 6; m(2,1) = 8;

				// partial transformation
				afb::TransformSecondTensorToVoigt(c, m, 3);

				Assert::AreEqual( 5, c(0), REAL_TOLERANCE);
				Assert::AreEqual( 2, c(1), REAL_TOLERANCE);
				Assert::AreEqual( 3, c(2), REAL_TOLERANCE);
				Assert::AreEqual( 8, c(3), REAL_TOLERANCE);
				Assert::AreEqual( 6, c(4), REAL_TOLERANCE);
				Assert::AreEqual(-4, c(5), REAL_TOLERANCE);
				Assert::AreEqual( 7, c(6), REAL_TOLERANCE);
			}
		}
		TEST_METHOD(TestVectorTransformTensorToVoigt_ErrorIncompatible)
		{
			afb::ColumnVector a(6), b(7), c(5), d(8);
			afb::SymmetricMatrix m(3), n(4);


			try
			{
				afb::TransformSecondTensorToVoigt(c, m);
				Assert::Fail(_T("Accepted incompatible vectors."));
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
				afb::TransformSecondTensorToVoigt(b, m);
				Assert::Fail(_T("Accepted incompatible vectors."));
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
				afb::TransformSecondTensorToVoigt(a, n, 3);
				Assert::Fail(_T("Accepted incompatible vectors."));
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
				afb::TransformSecondTensorToVoigt(d, n, 3);
				Assert::Fail(_T("Accepted incompatible vectors."));
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

		TEST_METHOD(TestMatrixVectorProduct_mv)
		{
			afb::DenseMatrix m(3,4);
			afb::ColumnVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) =  2; m(0,1) = 5; m(0,2) = 7; m(0,3) = 8;
			m(1,0) = -2; m(1,1) = 3; m(1,2) = 0; m(1,3) = 1;
			m(2,0) =  1; m(2,1) = 6; m(2,2) = 4; m(2,3) = 5;

			// ...and then vector
			v(0) = 3; v(1) = 1; v(2) = 2; v(3) = 7;

			// calculate product using first version			
			afb::VectorProduct(r, 1.0, m, afb::NotTransposed, v, afb::NotTransposed);

			// check result
			Assert::AreEqual((real)81.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real) 4.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)52.0, r(2), REAL_TOLERANCE);

			// calculate product using alternate version			
			r.ClearAll();
			afb::VectorProduct(r, 1.0, m, v);

			// check result
			Assert::AreEqual((real)81.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real) 4.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)52.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorProduct_mTv)
		{
			afb::DenseMatrix m(4,3);
			afb::ColumnVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) = 3; m(0,1) = 1; m(0,2) = 8;
			m(1,0) = 5; m(1,1) = 4; m(1,2) = 6;
			m(2,0) = 2; m(2,1) = 7; m(2,2) =-1;
			m(3,0) = 0; m(3,1) = 9; m(3,2) = 1;


			// ...and then vector
			v(0) = 5; v(1) = 4; v(2) = 9; v(3) = 8;

			// calculate product
			afb::VectorProduct(r, 1.0, m, afb::Transposed, v, afb::NotTransposed);

			// check result
			Assert::AreEqual((real) 53.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)156.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 63.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorProduct_mvT)
		{
			afb::DenseMatrix m(3,4);
			afb::RowVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) = 1; m(0,1) = 4; m(0,2) =-6; m(0,3) = 9;
			m(1,0) = 8; m(1,1) = 6; m(1,2) = 3; m(1,3) = 0;
			m(2,0) = 0; m(2,1) = 5; m(2,2) = 2; m(2,3) = 5;

			// ...and then vector
			v(0) = -2; v(1) = 4; v(2) = -5; v(3) = 9;

			// calculate product
			afb::VectorProduct(r, 1.0, m, afb::NotTransposed, v, afb::Transposed);

			// check result
			Assert::AreEqual((real)125.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real) -7.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 55.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorProduct_mTvT)
		{
			afb::DenseMatrix m(4,3);
			afb::RowVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) = 3; m(0,1) = 1; m(0,2) = 8;
			m(1,0) = 5; m(1,1) = 4; m(1,2) = 6;
			m(2,0) = 2; m(2,1) = 7; m(2,2) =-1;
			m(3,0) = 0; m(3,1) = 9; m(3,2) = 1;

			// ...and then vector
			v(0) = 5; v(1) = 4; v(2) = 9; v(3) = 8;

			// calculate product
			afb::VectorProduct(r, 1.0, m, afb::Transposed, v, afb::Transposed);

			// check result
			Assert::AreEqual((real) 53.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)156.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 63.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorProduct_ErrorIncompatible)
		{
			// check for dimension incompatibility between product factors
			afb::DenseMatrix m1(4,4), m2(5,3);
			afb::ColumnVector v1(5), v2(3);
			afb::ColumnVector r1(4), r2(5);
			try
			{
				afb::VectorProduct(r1, 1.0, m1, v1);
				Assert::Fail(_T("Incompatible dimension was not caught (m[4x4] * v[5x1])."));
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
				afb::VectorProduct(r2, 1.0, m2, afb::Transposed, v2, afb::NotTransposed);
				Assert::Fail(_T("Incompatible dimension was not caught (m[5,3]^T * v[3x1])."));
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

			// check for incompatibility in the result vector
			afb::DenseMatrix m3(4,4);
			afb::ColumnVector v3(4);
			afb::ColumnVector r3(5), r4(3);
			try
			{
				afb::VectorProduct(r3, 1.0, m3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x1 but accepted 5x1)."));
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
				afb::VectorProduct(r4, 1.0, m3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x1 but accepted 3x1)."));
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

		TEST_METHOD(TestMatrixVectorAccumulateProduct_vv)
		{
			afb::DenseMatrix m(3,4);
			afb::ColumnVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) =  2; m(0,1) = 5; m(0,2) = 7; m(0,3) = 8;
			m(1,0) = -2; m(1,1) = 3; m(1,2) = 0; m(1,3) = 1;
			m(2,0) =  1; m(2,1) = 6; m(2,2) = 4; m(2,3) = 5;

			// ...and then vectors
			v(0) = 3; v(1) = 1; v(2) = 2; v(3) = 7;
			r(0) = -8; r(1) = -6; r(2) = 4;

			// calculate product using first version
			afb::VectorAccumulateProduct(r, 1.0, m, afb::NotTransposed, v, afb::NotTransposed);

			// check result
			Assert::AreEqual((real)73.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-2.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)56.0, r(2), REAL_TOLERANCE);


			// calculate product
			r(0) = -8; r(1) = -6; r(2) = 4;
			afb::VectorAccumulateProduct(r, 1.0, m, v);

			// check result using alternate version
			Assert::AreEqual((real)73.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)-2.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real)56.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorAccumulateProduct_vTv)
		{
			afb::DenseMatrix m(4,3);
			afb::ColumnVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) = 3; m(0,1) = 1; m(0,2) = 8;
			m(1,0) = 5; m(1,1) = 4; m(1,2) = 6;
			m(2,0) = 2; m(2,1) = 7; m(2,2) =-1;
			m(3,0) = 0; m(3,1) = 9; m(3,2) = 1;


			// ...and then vectors
			v(0) = 5; v(1) = 4; v(2) = 9; v(3) = 8;
			r(0) = 3; r(1) = -7; r(2) = 4;

			// calculate product
			afb::VectorAccumulateProduct(r, 1.0, m, afb::Transposed, v, afb::NotTransposed);

			// check result
			Assert::AreEqual((real) 56.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)149.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 67.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorAccumulateProduct_vvT)
		{
			afb::DenseMatrix m(3,4);
			afb::RowVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) = 1; m(0,1) = 4; m(0,2) =-6; m(0,3) = 9;
			m(1,0) = 8; m(1,1) = 6; m(1,2) = 3; m(1,3) = 0;
			m(2,0) = 0; m(2,1) = 5; m(2,2) = 2; m(2,3) = 5;

			// ...and then vectors
			v(0) = -2; v(1) = 4; v(2) = -5; v(3) = 9;
			r(0) = 5; r(1) = -2; r(2) = -9; 

			// calculate product
			afb::VectorAccumulateProduct(r, 1.0, m, afb::NotTransposed, v, afb::Transposed);

			// check result
			Assert::AreEqual((real)130.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real) -9.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 46.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorAccumulateProduct_vTvT)
		{
			afb::DenseMatrix m(4,3);
			afb::RowVector v(4);
			afb::ColumnVector r(3);

			// populate matrix...
			m(0,0) = 3; m(0,1) = 1; m(0,2) = 8;
			m(1,0) = 5; m(1,1) = 4; m(1,2) = 6;
			m(2,0) = 2; m(2,1) = 7; m(2,2) =-1;
			m(3,0) = 0; m(3,1) = 9; m(3,2) = 1;

			// ...and then vectors
			v(0) = 5; v(1) = 4; v(2) = 9; v(3) = 8;
			r(0) = 1; r(1) = 2; r(2) = -1; 

			// calculate product
			afb::VectorAccumulateProduct(r, 1.0, m, afb::Transposed, v, afb::Transposed);

			// check result
			Assert::AreEqual((real) 54.0, r(0), REAL_TOLERANCE);
			Assert::AreEqual((real)158.0, r(1), REAL_TOLERANCE);
			Assert::AreEqual((real) 62.0, r(2), REAL_TOLERANCE);
		}
		TEST_METHOD(TestMatrixVectorAccumulateProduct_ErrorIncompatible)
		{
			// check for dimension incompatibility between product factors
			afb::DenseMatrix m1(4,4), m2(5,3);
			afb::ColumnVector v1(5), v2(3);
			afb::ColumnVector r1(4), r2(5);
			try
			{
				afb::VectorAccumulateProduct(r1, 1.0, m1, v1);
				Assert::Fail(_T("Incompatible dimension was not caught (m[4x4] * v[5x1])."));
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
				afb::VectorAccumulateProduct(r2, 1.0, m2, afb::Transposed, v2, afb::NotTransposed);
				Assert::Fail(_T("Incompatible dimension was not caught (m[5,3]^T * v[3x1])."));
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

			// check for incompatibility in the result vector
			afb::DenseMatrix m3(4,4);
			afb::ColumnVector v3(4);
			afb::ColumnVector r3(5), r4(3);
			afb::RowVector r5(4);
			try
			{
				afb::VectorAccumulateProduct(r3, 1.0, m3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x1 but accepted 5x1)."));
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
				afb::VectorAccumulateProduct(r4, 1.0, m3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x1 but accepted 3x1)."));
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
				afb::VectorAccumulateProduct(r5, 1.0, m3, v3);
				Assert::Fail(_T("Incompatible result dimension was not caught (should be 4x1 but accepted 1x4)."));
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
	};
}


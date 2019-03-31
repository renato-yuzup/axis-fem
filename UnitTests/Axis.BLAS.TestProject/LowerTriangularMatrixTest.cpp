#include "stdafx.h"
#include "foundation/blas/TriangularMatrix.hpp"
#include "foundation/DimensionMismatchException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "AxisString.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"

using namespace axis::foundation::blas;
using namespace axis;

// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const LowerTriangularMatrix& m)
{
	return std::wstring(_T("M[")) + String::int_parse(m.Rows()).data() + _T(",") + String::int_parse(m.Columns()).data() + _T("]");
}


namespace axis_blas_unit_tests
{
	TEST_CLASS(LowerTriangularMatrixTest)
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
		TEST_METHOD(BasicConstructorTest)
		{
			LowerTriangularMatrix m(5);
		}
		TEST_METHOD(AttributesTest)
		{
			LowerTriangularMatrix m(5);
			Assert::AreEqual(5L, m.Rows());
			Assert::AreEqual(5L, m.Columns());
			Assert::AreEqual(25L, m.ElementCount());
			Assert::AreEqual(15L, m.TriangularCount());

			LowerTriangularMatrix x(4);
			Assert::AreEqual(4L, x.Rows());
			Assert::AreEqual(4L, x.Columns());
			Assert::AreEqual(16L, x.ElementCount());
			Assert::AreEqual(10L, x.TriangularCount());
		}
		TEST_METHOD(AssignmentTest)
		{
			LowerTriangularMatrix m(3);
			m(0,0) = 5;
			m(1,0) = 3;
			m(1,1) = 7;
			m(2,0) = 2;
			m(2,1) = 6;
			m(2,2) = 9;

			Assert::AreEqual(5, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(7, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(6, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(9, m(2,2), REAL_TOLERANCE);

			// must not accept values that fall outside range
			try
			{
				real x = m(6, 2);
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
			try
			{
				m(7, 2) = 2;
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
		}
		TEST_METHOD(NullAreaTest)
		{
			LowerTriangularMatrix m(3);
			m(0,0) = 5;
			m(1,0) = 3;
			m(1,1) = 7;
			m(2,0) = 2;
			m(2,1) = 6;
			m(2,2) = 9;

			Assert::AreEqual(5, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(7, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(6, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(9, m(2,2), REAL_TOLERANCE);

			// null area must be zero, of course
			const LowerTriangularMatrix& x = m;
			Assert::AreEqual(0, x.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, x.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, x.GetElement(1,2), REAL_TOLERANCE);

			// any assignment to null area must throw an exception
			try
			{
				m(0,2) = 4;
				Assert::Fail(_T("Exception not thrown when changing null area of triangular matrix."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
		}
		TEST_METHOD(GetterSetterTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 2);
			m.SetElement(1, 1, 4);
			Assert::AreEqual(1, m.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m.GetElement(1,1), REAL_TOLERANCE);

			// must not accept values that fall outside range
			try
			{
				real x = m.GetElement(6, 2);
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
			try
			{
				m.SetElement(1, 7, 2);
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
			try
			{	// ...and also must not accept in the null area
				m.SetElement(0, 1, 2);
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
		}
		TEST_METHOD(CopyConstructorTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 9);
			m.SetElement(1, 1, 4);

			// copy matrix
			LowerTriangularMatrix c(m);
			Assert::AreEqual(2L, c.Rows());
			Assert::AreEqual(2L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, c.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, c.GetElement(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(CopyAssignmentTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 9);
			m.SetElement(1, 1, 4);

			// copy LowerTriangularMatrix
			LowerTriangularMatrix c(2);
			c = m;
			Assert::AreEqual(2L, c.Rows());
			Assert::AreEqual(2L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, c.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, c.GetElement(1,1), REAL_TOLERANCE);

			// this test must fail
			try
			{
				LowerTriangularMatrix r(3);
				m = r;
				Assert::Fail(_T("Unexpected copy assignment behavior; matrices of different sizes were accepted."));
			}
			catch (axis::foundation::DimensionMismatchException)
			{
				// test ok
			}
		}
		TEST_METHOD(ConstructorFullInitializeVectorTest)
		{
			// build an array which contains the whole matrix data
			real * data = new real[6];
			data[0] = -1;
			data[1] = 3;
			data[2] = 1;
			data[3] = 2;
			data[4] = 5;
			data[5] = 4;

			LowerTriangularMatrix m(3,data);
			delete [] data;

			Assert::AreEqual(-1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(1,2), REAL_TOLERANCE);
			Assert::AreEqual(2, m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(5, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(4, m(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ConstructorPartialInitializeVectorTest)
		{
			// build an array which contains part of matrix data
			real * data = new real[4];
			data[0] = -1;
			data[1] = 3;
			data[2] = 1;
			data[3] = 2;

			LowerTriangularMatrix m(3, data, 4);
			delete [] data;

			Assert::AreEqual(-1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(1,2), REAL_TOLERANCE);
			Assert::AreEqual(2, m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ConstructorSetAllTest)
		{
			LowerTriangularMatrix m(3, 8);
			Assert::AreEqual(8, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(8, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(8, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(1,2), REAL_TOLERANCE);
			Assert::AreEqual(8, m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(8, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(8, m(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ClearAllTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, -5);
			m.SetElement(1, 1, 4);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			m.ClearAll();
			Assert::AreEqual(0, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(SetAllTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 5);
			m.SetElement(1, 1, 4);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(5, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			m.SetAll(2);
			Assert::AreEqual(2, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(IncrementElementTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 2);
			m.SetElement(1, 1, 4);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			m.Accumulate(0, 0, 7).Accumulate(1,0, -1);
			Assert::AreEqual(8, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(1, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);

			// this one must fail
			try
			{
				m.Accumulate(5, 0, 1);
				Assert::Fail(_T("Accumulate element out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
			// this one must fail too
			try
			{
				m.Accumulate(0, 1, 4);
				Assert::Fail(_T("Accumulate element on prohibited area did not throw an exception."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
		}
		TEST_METHOD(IncrementLowerTriangularMatrixTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 3);
			m.SetElement(1, 1, 2);

			LowerTriangularMatrix x(2);
			x.SetElement(0, 0, 3);
			x.SetElement(1, 0, 5);
			x.SetElement(1, 1, 4);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m += x));

			Assert::AreEqual(4, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(8, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(6, m(1,1), REAL_TOLERANCE);

			Assert::AreEqual(3, x(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, x.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(5, x(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, x(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(DecrementLowerTriangularMatrixTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 5);
			m.SetElement(1, 1, 4);

			LowerTriangularMatrix x(2);
			x.SetElement(0, 0, 6);
			x.SetElement(1, 0, 8);
			x.SetElement(1, 1, 7);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m -= x));

			Assert::AreEqual(-5, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-3, m(1,1), REAL_TOLERANCE);

			Assert::AreEqual(6, x(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, x.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(8, x(1,0), REAL_TOLERANCE);
			Assert::AreEqual(7, x(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarMultiplyOperatorTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 4);
			m.SetElement(1, 0, 3);
			m.SetElement(1, 1, -5);
			Assert::AreEqual(4, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m *= 10));

			Assert::AreEqual(40, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(30, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-50, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScaleMethodTest)
		{
			LowerTriangularMatrix m(2, 2);
			m.SetElement(0, 0, 4);
			m.SetElement(1, 0, 3);
			m.SetElement(1, 1, -5);
			Assert::AreEqual(4, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (LowerTriangularMatrix&)m.Scale(10));

			Assert::AreEqual(40, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(30, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-50, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarDivideOperatorTest)
		{
			LowerTriangularMatrix m(2);
			m.SetElement(0, 0, 7);
			m.SetElement(1, 0, 9);
			m.SetElement(1, 1, -3);
			Assert::AreEqual(7, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(9, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-3, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m /= 4));

			Assert::AreEqual((real)1.75, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual((real)0, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual((real)2.25, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual((real)-0.75, m(1,1)), REAL_TOLERANCE;
		}
		TEST_METHOD(TraceTest)
		{
			LowerTriangularMatrix m(3,3);
			m(0,0) = 2;
			m(1,1) = 5;
			m(2,2) = 1;

			real t = m.Trace();
			Assert::AreEqual((real)8.0, t);
		}
		TEST_METHOD(CopyFromTest)
		{
			LowerTriangularMatrix m(3,3), r(3,3);
			m(0,0) = 1; m(1,1) = 3;
			m(2,2) = 2; m(2,1) = 4;
			m(2,0) = 9; m(1,0) = 1;

			r.CopyFrom(m);

			Assert::AreEqual(1, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, r(2,2), REAL_TOLERANCE);
			Assert::AreEqual(4, r(2,1), REAL_TOLERANCE);
			Assert::AreEqual(9, r(2,0), REAL_TOLERANCE);
			Assert::AreEqual(1, r(1,0), REAL_TOLERANCE);
		}
    TEST_METHOD(AllocateFromModelMemoryTest)
    {
      axis::foundation::memory::RelativePointer ptr1 = 
        LowerTriangularMatrix::Create(2);
      axis::foundation::memory::RelativePointer ptr2 = 
        LowerTriangularMatrix::Create(2);

      LowerTriangularMatrix& m1 = *(LowerTriangularMatrix *)*ptr1;
      LowerTriangularMatrix& m2 = *(LowerTriangularMatrix *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &m1 == &m2);

      m1(0,0) = 10; m1(1,0) = -8; m1(1,1) = 5;
      m2(0,0) = 1;  m2(1,0) = 6;  m2(1,1) = 4;

      m1 += m2;
      Assert::AreEqual(11, m1.GetElement(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 0, m1.GetElement(0,1), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1.GetElement(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 9, m1.GetElement(1,1), REAL_TOLERANCE);
      Assert::AreEqual( 1, m2.GetElement(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 0, m2.GetElement(0,1), REAL_TOLERANCE);
      Assert::AreEqual( 6, m2.GetElement(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 4, m2.GetElement(1,1), REAL_TOLERANCE);

      m1.Destroy();
      m2.Destroy();
      System::ModelMemory().Deallocate(ptr1);
      System::ModelMemory().Deallocate(ptr2);
    }
    TEST_METHOD(AllocateFromGlobalMemoryTest)
    {
      axis::foundation::memory::RelativePointer ptr1 = 
        LowerTriangularMatrix::CreateFromGlobalMemory(2);
      axis::foundation::memory::RelativePointer ptr2 = 
        LowerTriangularMatrix::CreateFromGlobalMemory(2);

      LowerTriangularMatrix& m1 = *(LowerTriangularMatrix *)*ptr1;
      LowerTriangularMatrix& m2 = *(LowerTriangularMatrix *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &m1 == &m2);

      m1(0,0) = 10; m1(1,0) = -8; m1(1,1) = 5;
      m2(0,0) = 1;  m2(1,0) = 6;  m2(1,1) = 4;

      m1 += m2;
      Assert::AreEqual(11, m1.GetElement(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 0, m1.GetElement(0,1), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1.GetElement(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 9, m1.GetElement(1,1), REAL_TOLERANCE);
      Assert::AreEqual( 1, m2.GetElement(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 0, m2.GetElement(0,1), REAL_TOLERANCE);
      Assert::AreEqual( 6, m2.GetElement(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 4, m2.GetElement(1,1), REAL_TOLERANCE);

      m1.Destroy();
      m2.Destroy();
      System::GlobalMemory().Deallocate(ptr1);
      System::GlobalMemory().Deallocate(ptr2);
    }
	};
}



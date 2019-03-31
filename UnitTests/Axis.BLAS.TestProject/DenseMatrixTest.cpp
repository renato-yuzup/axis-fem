#include "stdafx.h"
#include "foundation/blas/DenseMatrix.hpp"
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
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const DenseMatrix& m)
{
	return std::wstring(_T("M[")) + String::int_parse(m.Rows()).data() + _T(",") + String::int_parse(m.Columns()).data() + _T("]");
}


namespace axis_blas_unit_tests
{
	TEST_CLASS(DenseMatrixTest)
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
			DenseMatrix m(5,5);
		}
		TEST_METHOD(AttributesTest)
		{
			DenseMatrix m(5,7);
			Assert::AreEqual(5L, m.Rows());
			Assert::AreEqual(7L, m.Columns());
			Assert::AreEqual(35L, m.ElementCount());
			Assert::AreEqual(false, m.IsSquare());

			DenseMatrix x(4,4);
			Assert::AreEqual(4L, x.Rows());
			Assert::AreEqual(4L, x.Columns());
			Assert::AreEqual(16L, x.ElementCount());
			Assert::AreEqual(true, x.IsSquare());
		}
		TEST_METHOD(AssignmentTest)
		{
			DenseMatrix m(4,3);
			m(0,0) = -1; m(0,1) = +3; m(0,2) = +1;
			m(1,0) = +2; m(1,1) = +1; m(1,2) = +4;
			m(2,0) = +7; m(2,1) = -5; m(2,2) = +9;
			m(3,0) = +5; m(3,1) = +6; m(3,2) = +8;

			Assert::AreEqual(-1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3,  m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(1,  m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(2,  m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(1,  m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(4,  m(1,2), REAL_TOLERANCE);
			Assert::AreEqual(7,  m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(9,  m(2,2), REAL_TOLERANCE);
			Assert::AreEqual(5,  m(3,0), REAL_TOLERANCE);
			Assert::AreEqual(6,  m(3,1), REAL_TOLERANCE);
			Assert::AreEqual(8,  m(3,2), REAL_TOLERANCE);

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
		TEST_METHOD(GetterSetterTest)
		{
			DenseMatrix m(2, 3);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(0, 2, 7);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);
			m.SetElement(1, 2, 2);
			Assert::AreEqual(1, m.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, m.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(7, m.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m.GetElement(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m.GetElement(1,2), REAL_TOLERANCE);

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
		}
		TEST_METHOD(CopyConstructorTest)
		{
			DenseMatrix m(2, 3);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(0, 2, 7);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);
			m.SetElement(1, 2, 2);

			// copy matrix
			DenseMatrix c(m);
			Assert::AreEqual(2L, c.Rows());
			Assert::AreEqual(3L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(7, c.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, c.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, c.GetElement(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, c.GetElement(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(CopyAssignmentTest)
		{
			DenseMatrix m(2, 3);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(0, 2, 7);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);
			m.SetElement(1, 2, 2);

			// copy matrix
			DenseMatrix c(2, 3);
			c = m;
			Assert::AreEqual(2L, c.Rows());
			Assert::AreEqual(3L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(7, c.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, c.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, c.GetElement(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, c.GetElement(1,2), REAL_TOLERANCE);

			// this test must fail
			try
			{
				DenseMatrix r(2,2);
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
			real * data = new real[12];
			data[0] = -1;
			data[1] = 3;
			data[2] = 1;
			data[3] = 2;
			data[4] = 1;
			data[5] = 4;
			data[6] = 7;
			data[7] = -5;
			data[8] = 9;
			data[9] = 5;
			data[10] = 6;
			data[11] = 8;

			DenseMatrix m(4, 3, data);
			delete [] data;

			Assert::AreEqual(-1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3,  m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(1,  m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(2,  m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(1,  m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(4,  m(1,2), REAL_TOLERANCE);
			Assert::AreEqual(7,  m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(9,  m(2,2), REAL_TOLERANCE);
			Assert::AreEqual(5,  m(3,0), REAL_TOLERANCE);
			Assert::AreEqual(6,  m(3,1), REAL_TOLERANCE);
			Assert::AreEqual(8,  m(3,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ConstructorPartialInitializeVectorTest)
		{
			// build an array which contains part of matrix data
			real * data = new real[4];
			data[0] = -1;
			data[1] = 3;
			data[2] = 1;
			data[3] = 2;

			DenseMatrix m(4, 3, data, 4);
			delete [] data;

			Assert::AreEqual(-1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3,  m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(1,  m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(2,  m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(1,2), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(2,2), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(3,0), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(3,1), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(3,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ConstructorSetAllTest)
		{
			DenseMatrix m(2, 3, 4);
			Assert::AreEqual(4, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(4, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ClearAllTest)
		{
			DenseMatrix m(2, 3);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(0, 2, 7);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);
			m.SetElement(1, 2, 2);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(7, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,2), REAL_TOLERANCE);
			m.ClearAll();
			Assert::AreEqual(0, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(SetAllTest)
		{
			DenseMatrix m(2, 3);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(0, 2, 7);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);
			m.SetElement(1, 2, 2);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(7, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,2), REAL_TOLERANCE);
			m.SetAll(2);
			Assert::AreEqual(2, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,2), REAL_TOLERANCE);
		}
		TEST_METHOD(IncrementElementTest)
		{
			DenseMatrix m(2, 2);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			m.Accumulate(0, 0, 7).Accumulate(0,1, -2);
			Assert::AreEqual(8, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(7, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,0), REAL_TOLERANCE);
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
		}
		TEST_METHOD(IncrementMatrixTest)
		{
			DenseMatrix m(2, 2);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);

			DenseMatrix x(2, 2);
			x.SetElement(0, 0, 0);
			x.SetElement(0, 1, 3);
			x.SetElement(1, 0, -1);
			x.SetElement(1, 1, 2);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m += x));

			Assert::AreEqual(1,  m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(12, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-1, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(6,  m(1,1), REAL_TOLERANCE);

			Assert::AreEqual(0,  x(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3,  x(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-1, x(1,0), REAL_TOLERANCE);
			Assert::AreEqual(2,  x(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(DecrementMatrixTest)
		{
			DenseMatrix m(2, 2);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, 4);

			DenseMatrix x(2, 2);
			x.SetElement(0, 0, 1);
			x.SetElement(0, 1, 3);
			x.SetElement(1, 0, -1);
			x.SetElement(1, 1, 2);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m -= x));

			Assert::AreEqual(0, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(6, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(1, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,1), REAL_TOLERANCE);

			Assert::AreEqual(1,  x(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3,  x(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-1, x(1,0), REAL_TOLERANCE);
			Assert::AreEqual(2,  x(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarMultiplyOperatorTest)
		{
			DenseMatrix m(2, 2);
			m.SetElement(0, 0, 4);
			m.SetElement(0, 1, 3);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, -5);
			Assert::AreEqual(4,  m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3,  m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m *= 10));

			Assert::AreEqual(40,  m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(30,  m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0,   m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-50, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScaleMethodTest)
		{
			DenseMatrix m(2, 2);
			m.SetElement(0, 0, 4);
			m.SetElement(0, 1, 3);
			m.SetElement(1, 0, 0);
			m.SetElement(1, 1, -5);
			Assert::AreEqual(4,  m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3,  m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (DenseMatrix&)m.Scale(10));

			Assert::AreEqual(40,  m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(30,  m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0,   m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-50, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarDivideOperatorTest)
		{
			DenseMatrix m(2, 2);
			m.SetElement(0, 0, 8);
			m.SetElement(0, 1, 0);
			m.SetElement(1, 0, 6);
			m.SetElement(1, 1, -12);
			Assert::AreEqual(8,   m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0,   m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(6,   m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-12, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m /= 4));

			Assert::AreEqual(2,   m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0,   m(0,1), REAL_TOLERANCE);
			Assert::AreEqual((real)1.5, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-3,  m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TraceTest)
		{
			DenseMatrix m(3,3);
			m(0,0) = 2;
			m(1,1) = 5;
			m(2,2) = 1;

			real t = m.Trace();
			Assert::AreEqual((real)8.0, t);
		}
		TEST_METHOD(CopyFromTest)
		{
			DenseMatrix m(2,2), r(2,2);
			m(0,0) = 1; m(0,1) = 7;
			m(1,0) = 10; m(1,1) = 4;

			r.CopyFrom(m);

			Assert::AreEqual( 1, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual( 7, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual(10, r(1,0), REAL_TOLERANCE);
			Assert::AreEqual( 4, r(1,1), REAL_TOLERANCE);
		}
    TEST_METHOD(AllocateFromModelMemoryTest)
    {
      axis::foundation::memory::RelativePointer ptr1 = 
        DenseMatrix::Create(2,2);
      axis::foundation::memory::RelativePointer ptr2 = 
        DenseMatrix::Create(2,2);

      DenseMatrix& m1 = *(DenseMatrix *)*ptr1;
      DenseMatrix& m2 = *(DenseMatrix *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &m1 == &m2);

      m1(0,0) = 10; m1(0,1) = 3; m1(1,0) = -8; m1(1,1) = 5;
      m2(0,0) = 1; m2(0,1) = 2; m2(1,0) = 6; m2(1,1) = 4;

      m1 += m2;
      Assert::AreEqual(11, m1(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 5, m1(0,1), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 9, m1(1,1), REAL_TOLERANCE);
      Assert::AreEqual( 1, m2(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 2, m2(0,1), REAL_TOLERANCE);
      Assert::AreEqual( 6, m2(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 4, m2(1,1), REAL_TOLERANCE);

      m1.Destroy();
      m2.Destroy();
      System::ModelMemory().Deallocate(ptr1);
      System::ModelMemory().Deallocate(ptr2);
    }
    TEST_METHOD(AllocateFromGlobalMemoryTest)
    {
      axis::foundation::memory::RelativePointer ptr1 = 
        DenseMatrix::CreateFromGlobalMemory(2,2);
      axis::foundation::memory::RelativePointer ptr2 = 
        DenseMatrix::CreateFromGlobalMemory(2,2);

      DenseMatrix& m1 = *(DenseMatrix *)*ptr1;
      DenseMatrix& m2 = *(DenseMatrix *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &m1 == &m2);

      m1(0,0) = 10; m1(0,1) = 3; m1(1,0) = -8; m1(1,1) = 5;
      m2(0,0) = 1; m2(0,1) = 2; m2(1,0) = 6; m2(1,1) = 4;

      m1 += m2;
      Assert::AreEqual(11, m1(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 5, m1(0,1), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 9, m1(1,1), REAL_TOLERANCE);
      Assert::AreEqual( 1, m2(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 2, m2(0,1), REAL_TOLERANCE);
      Assert::AreEqual( 6, m2(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 4, m2(1,1), REAL_TOLERANCE);

      m1.Destroy();
      m2.Destroy();
      System::GlobalMemory().Deallocate(ptr1);
      System::GlobalMemory().Deallocate(ptr2);
    }
	};
}


#include "stdafx.h"

#include "foundation/blas/SymmetricMatrix.hpp"
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
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const SymmetricMatrix& m)
{
	return std::wstring(_T("M[")) + String::int_parse(m.Rows()).data() + _T(",") + String::int_parse(m.Columns()).data() + _T("]");
}


namespace axis_blas_unit_tests
{
	TEST_CLASS(SymmetricMatrixTest)
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
			SymmetricMatrix m(5);
		}
		TEST_METHOD(AttributesTest)
		{
			SymmetricMatrix m(5);
			Assert::AreEqual(5L, m.Rows());
			Assert::AreEqual(5L, m.Columns());
			Assert::AreEqual(25L, m.ElementCount());
			Assert::AreEqual(15L, m.SymmetricCount());

			SymmetricMatrix x(4);
			Assert::AreEqual(4L, x.Rows());
			Assert::AreEqual(4L, x.Columns());
			Assert::AreEqual(16L, x.ElementCount());
			Assert::AreEqual(10L, x.SymmetricCount());
		}
		TEST_METHOD(AssignmentTest)
		{
			SymmetricMatrix m(3);
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
		TEST_METHOD(SymmetricAssignmentTest)
		{
			SymmetricMatrix m(3);
			m(0,0) = 5;
			m(1,0) = 3;
			m(1,1) = 7;
			m(2,0) = 2;
			m(2,1) = 6;
			m(2,2) = 9;

			Assert::AreEqual(5, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(7, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(6, m(1,2), REAL_TOLERANCE);
			Assert::AreEqual(9, m(2,2), REAL_TOLERANCE);

			// now, modify in symmetric elements; an overwrite is expected
			m(1, 2) = -5;
			Assert::AreEqual(-5, m(1, 2), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(2, 1), REAL_TOLERANCE);
			m(2, 1) = 8;
			Assert::AreEqual(8, m(1, 2), REAL_TOLERANCE);
			Assert::AreEqual(8, m(2, 1), REAL_TOLERANCE);
		}
		TEST_METHOD(GetterSetterTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 2);
			m.SetElement(1, 1, 4);
			Assert::AreEqual(1, m.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m.GetElement(0,1), REAL_TOLERANCE);
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
		}
		TEST_METHOD(CopyConstructorTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(1, 1, 4);

			// copy matrix
			SymmetricMatrix c(m);
			Assert::AreEqual(2L, c.Rows());
			Assert::AreEqual(2L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, c.GetElement(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(CopyAssignmentTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 9);
			m.SetElement(1, 1, 4);

			// copy SymmetricMatrix
			SymmetricMatrix c(2);
			c = m;
			Assert::AreEqual(2L, c.Rows());
			Assert::AreEqual(2L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, c.GetElement(1,1), REAL_TOLERANCE);

			// this test must fail
			try
			{
				SymmetricMatrix r(3);
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

			SymmetricMatrix m(3,data);
			delete [] data;

			Assert::AreEqual(-1,m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(5, m(1,2), REAL_TOLERANCE);
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

			SymmetricMatrix m(3, data, 4);
			delete [] data;

			Assert::AreEqual(-1,m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,2), REAL_TOLERANCE);
			Assert::AreEqual(2, m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ConstructorSetAllTest)
		{
			SymmetricMatrix m(3, 8);
			Assert::AreEqual(8, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(8, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(8, m(0,2), REAL_TOLERANCE);
			Assert::AreEqual(8, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(8, m(1,1), REAL_TOLERANCE);
			Assert::AreEqual(8, m(1,2), REAL_TOLERANCE);
			Assert::AreEqual(8, m(2,0), REAL_TOLERANCE);
			Assert::AreEqual(8, m(2,1), REAL_TOLERANCE);
			Assert::AreEqual(8, m(2,2), REAL_TOLERANCE);
		}
		TEST_METHOD(ClearAllTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, -5);
			m.SetElement(1, 1, 4);

			Assert::AreEqual(1,  m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4,  m(1,1), REAL_TOLERANCE);
			m.ClearAll();
			Assert::AreEqual(0, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(SetAllTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 5);
			m.SetElement(1, 1, 4);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(5, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(5, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			m.SetAll(2);
			Assert::AreEqual(2, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(IncrementElementTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 2);
			m.SetElement(1, 1, 4);

			Assert::AreEqual(1, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, m(1,1), REAL_TOLERANCE);
			m.Accumulate(0, 0, 7).Accumulate(0,1, -2);
			Assert::AreEqual(8, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(0,1), REAL_TOLERANCE);
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
		TEST_METHOD(IncrementSymmetricMatrixTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(1, 0, 3);
			m.SetElement(1, 1, 2);

			SymmetricMatrix x(2);
			x.SetElement(0, 0, 3);
			x.SetElement(1, 0, 5);
			x.SetElement(1, 1, 4);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m += x));

			Assert::AreEqual(4, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(8, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(8, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(6, m(1,1), REAL_TOLERANCE);

			Assert::AreEqual(3, x(0,0), REAL_TOLERANCE);
			Assert::AreEqual(5, x(0,1), REAL_TOLERANCE);
			Assert::AreEqual(5, x(1,0), REAL_TOLERANCE);
			Assert::AreEqual(4, x(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(DecrementSymmetricMatrixTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 1);
			m.SetElement(0, 1, 5);
			m.SetElement(1, 1, 4);

			SymmetricMatrix x(2);
			x.SetElement(0, 0, 6);
			x.SetElement(0, 1, 8);
			x.SetElement(1, 1, 7);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m -= x));

			Assert::AreEqual(-5, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(-3, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(-3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-3, m(1,1), REAL_TOLERANCE);

			Assert::AreEqual(6, x(0,0), REAL_TOLERANCE);
			Assert::AreEqual(8, x(0,1), REAL_TOLERANCE);
			Assert::AreEqual(8, x(1,0), REAL_TOLERANCE);
			Assert::AreEqual(7, x(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarMultiplyOperatorTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 4);
			m.SetElement(0, 1, 3);
			m.SetElement(1, 1, -5);
			Assert::AreEqual(4, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m *= 10));

			Assert::AreEqual(40, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(30, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(30, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-50, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScaleMethodTest)
		{
			SymmetricMatrix m(2, 2);
			m.SetElement(0, 0, 4);
			m.SetElement(0, 1, 3);
			m.SetElement(1, 1, -5);
			Assert::AreEqual(4, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-5, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (SymmetricMatrix&)m.Scale(10));

			Assert::AreEqual(40, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(30, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(30, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-50, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarDivideOperatorTest)
		{
			SymmetricMatrix m(2);
			m.SetElement(0, 0, 7);
			m.SetElement(0, 1, 9);
			m.SetElement(1, 1, -3);
			Assert::AreEqual(7, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual(9, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual(9, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual(-3, m(1,1), REAL_TOLERANCE);

			// we must ensure that the same matrix is being returned by
			// this statement
			Assert::AreSame(m, (m /= 4));

			Assert::AreEqual((real)1.75, m(0,0), REAL_TOLERANCE);
			Assert::AreEqual((real)2.25, m(0,1), REAL_TOLERANCE);
			Assert::AreEqual((real)2.25, m(1,0), REAL_TOLERANCE);
			Assert::AreEqual((real)-0.75, m(1,1), REAL_TOLERANCE);
		}
		TEST_METHOD(TraceTest)
		{
			SymmetricMatrix m(3,3);
			m(0,0) = 2;
			m(1,1) = 5;
			m(2,2) = 1;

			real t = m.Trace();
			Assert::AreEqual((real)8.0, t);
		}
		TEST_METHOD(CopyFromTest)
		{
			SymmetricMatrix m(3,3), r(3,3);
			m(0,0) = 1; m(1,1) = 7;
			m(2,2) = 3; m(1,2) = 2;
			m(0,1) = 5; m(0,2) = 6;

			r.CopyFrom(m);

			Assert::AreEqual((real)1.0, r(0,0), REAL_TOLERANCE);
			Assert::AreEqual((real)7.0, r(1,1), REAL_TOLERANCE);
			Assert::AreEqual((real)3.0, r(2,2), REAL_TOLERANCE);
			Assert::AreEqual((real)2.0, r(1,2), REAL_TOLERANCE);
			Assert::AreEqual((real)5.0, r(0,1), REAL_TOLERANCE);
			Assert::AreEqual((real)6.0, r(0,2), REAL_TOLERANCE);
		}
    TEST_METHOD(AllocateFromModelMemoryTest)
    {
      axis::foundation::memory::RelativePointer ptr1 = 
        SymmetricMatrix::Create(2);
      axis::foundation::memory::RelativePointer ptr2 = 
        SymmetricMatrix::Create(2);

      SymmetricMatrix& m1 = *(SymmetricMatrix *)*ptr1;
      SymmetricMatrix& m2 = *(SymmetricMatrix *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &m1 == &m2);

      m1(0,0) = 10; m1(1,0) = -8; m1(1,1) = 5;
      m2(0,0) = 1;  m2(1,0) = 6;  m2(1,1) = 4;

      m1 += m2;
      Assert::AreEqual(11, m1(0,0), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1(0,1), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 9, m1(1,1), REAL_TOLERANCE);
      Assert::AreEqual( 1, m2(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 6, m2(0,1), REAL_TOLERANCE);
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
        SymmetricMatrix::CreateFromGlobalMemory(2);
      axis::foundation::memory::RelativePointer ptr2 = 
        SymmetricMatrix::CreateFromGlobalMemory(2);

      SymmetricMatrix& m1 = *(SymmetricMatrix *)*ptr1;
      SymmetricMatrix& m2 = *(SymmetricMatrix *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &m1 == &m2);

      m1(0,0) = 10; m1(1,0) = -8; m1(1,1) = 5;
      m2(0,0) = 1;  m2(1,0) = 6;  m2(1,1) = 4;

      m1 += m2;
      Assert::AreEqual(11, m1(0,0), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1(0,1), REAL_TOLERANCE);
      Assert::AreEqual(-2, m1(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 9, m1(1,1), REAL_TOLERANCE);
      Assert::AreEqual( 1, m2(0,0), REAL_TOLERANCE);
      Assert::AreEqual( 6, m2(0,1), REAL_TOLERANCE);
      Assert::AreEqual( 6, m2(1,0), REAL_TOLERANCE);
      Assert::AreEqual( 4, m2(1,1), REAL_TOLERANCE);

      m1.Destroy();
      m2.Destroy();
      System::GlobalMemory().Deallocate(ptr1);
      System::GlobalMemory().Deallocate(ptr2);
    }
	};
}



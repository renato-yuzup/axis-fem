#include "stdafx.h"
#include <math.h>
#include "foundation/blas/RowVector.hpp"
#include "foundation/blas/ColumnVector.hpp"
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
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const RowVector& v)
{
	return std::wstring(_T("v[")) + String::int_parse(v.Length()).data() + _T("]");
}


namespace axis_blas_unit_tests
{
	TEST_CLASS(RowVectorTest)
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
			RowVector m(5);
		}
		TEST_METHOD(AttributesTest)
		{
			RowVector m(5);
			Assert::AreEqual(1L, m.Rows());
			Assert::AreEqual(5L, m.Columns());
			Assert::AreEqual(5L, m.Length());

			RowVector x(4);
			Assert::AreEqual(1L, x.Rows());
			Assert::AreEqual(4L, x.Columns());
			Assert::AreEqual(4L, x.Length());
		}
		TEST_METHOD(AssignmentTest)
		{
			RowVector m(4);
			m(0) = -1; m(1) = 40; m(2) = 100; m(3) = -10;

			Assert::AreEqual(-1,  m(0), REAL_TOLERANCE);
			Assert::AreEqual(40,  m(1), REAL_TOLERANCE);
			Assert::AreEqual(100, m(2), REAL_TOLERANCE);
			Assert::AreEqual(-10, m(3), REAL_TOLERANCE);

			// must not accept values that fall outside range
			try
			{
				real x = m(6);
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
			try
			{
				m(-2) = 2;
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
		}
		TEST_METHOD(GetterSetterTest)
		{
			RowVector m(3);
			m.SetElement(0, 1);
			m.SetElement(1, 9);
			m.SetElement(2, 7);
			Assert::AreEqual(1, m.GetElement(0), REAL_TOLERANCE);
			Assert::AreEqual(9, m.GetElement(1), REAL_TOLERANCE);
			Assert::AreEqual(7, m.GetElement(2), REAL_TOLERANCE);

			// must not accept values that fall outside range
			try
			{
				real x = m.GetElement(6);
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
			try
			{
				m.SetElement(7, 2);
				Assert::Fail(_T("Out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
		}
		TEST_METHOD(CopyConstructorTest)
		{
			RowVector m(4);
			m.SetElement(0, 1);
			m.SetElement(1, 9);
			m.SetElement(2, 7);
			m.SetElement(3, 0);

			// copy vector
			RowVector c(m);
			Assert::AreEqual(1L, c.Rows());
			Assert::AreEqual(4L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(1), REAL_TOLERANCE);
			Assert::AreEqual(7, c.GetElement(2), REAL_TOLERANCE);
			Assert::AreEqual(0, c.GetElement(3), REAL_TOLERANCE);
		}
		TEST_METHOD(CopyAssignmentTest)
		{
			RowVector m(3);
			m.SetElement(0, 1);
			m.SetElement(1, 9);
			m.SetElement(2, 7);

			// copy vector
			RowVector c(3);
			c = m;
			Assert::AreEqual(1L, c.Rows());
			Assert::AreEqual(3L, c.Columns());

			Assert::AreEqual(1, c.GetElement(0), REAL_TOLERANCE);
			Assert::AreEqual(9, c.GetElement(1), REAL_TOLERANCE);
			Assert::AreEqual(7, c.GetElement(2), REAL_TOLERANCE);

			// this test must fail
			try
			{
				RowVector r(2);
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
			// build an array which contains the whole vector data
			real * data = new real[4];
			data[0] = -1;
			data[1] = 3;
			data[2] = 1;
			data[3] = 2;

			RowVector m(4, data);
			delete [] data;

			Assert::AreEqual(-1, m(0), REAL_TOLERANCE);
			Assert::AreEqual(3,  m(1), REAL_TOLERANCE);
			Assert::AreEqual(1,  m(2), REAL_TOLERANCE);
			Assert::AreEqual(2,  m(3), REAL_TOLERANCE);
		}
		TEST_METHOD(ConstructorPartialInitializeVectorTest)
		{
			// build an array which contains part of vector data
			real * data = new real[2];
			data[0] = -1;
			data[1] = 3;

			RowVector m(4, data, 2);
			delete [] data;

			Assert::AreEqual(-1, m(0), REAL_TOLERANCE);
			Assert::AreEqual(3,  m(1), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(2), REAL_TOLERANCE);
			Assert::AreEqual(0,  m(3), REAL_TOLERANCE);
		}
		TEST_METHOD(ConstructorSetAllTest)
		{
			RowVector m(3, 1);
			Assert::AreEqual(1, m(0), REAL_TOLERANCE);
			Assert::AreEqual(1, m(1), REAL_TOLERANCE);
			Assert::AreEqual(1, m(2), REAL_TOLERANCE);
		}
		TEST_METHOD(ClearAllTest)
		{
			RowVector m(3);
			m.SetElement(0, 1);
			m.SetElement(1, 9);
			m.SetElement(2, 7);

			Assert::AreEqual(1, m(0), REAL_TOLERANCE);
			Assert::AreEqual(9, m(1), REAL_TOLERANCE);
			Assert::AreEqual(7, m(2), REAL_TOLERANCE);
			m.ClearAll();
			Assert::AreEqual(0, m(0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1), REAL_TOLERANCE);
			Assert::AreEqual(0, m(2), REAL_TOLERANCE);
		}
		TEST_METHOD(SetAllTest)
		{
			RowVector m(3);
			m.SetElement(0, 1);
			m.SetElement(1, 9);
			m.SetElement(2, 7);

			Assert::AreEqual(1, m(0), REAL_TOLERANCE);
			Assert::AreEqual(9, m(1), REAL_TOLERANCE);
			Assert::AreEqual(7, m(2), REAL_TOLERANCE);
			m.SetAll(2);
			Assert::AreEqual(2, m(0), REAL_TOLERANCE);
			Assert::AreEqual(2, m(1), REAL_TOLERANCE);
			Assert::AreEqual(2, m(2), REAL_TOLERANCE);
		}
		TEST_METHOD(IncrementElementTest)
		{
			RowVector m(2);
			m.SetElement(0, 1);
			m.SetElement(1, 9);

			Assert::AreEqual(1, m(0), REAL_TOLERANCE);
			Assert::AreEqual(9, m(1), REAL_TOLERANCE);
			m.Accumulate(0, 7).Accumulate(1, -2);
			Assert::AreEqual(8, m(0), REAL_TOLERANCE);
			Assert::AreEqual(7, m(1), REAL_TOLERANCE);

			// this one must fail
			try
			{
				m.Accumulate(5, 1);
				Assert::Fail(_T("Accumulate element out of bounds error not thrown."));
			}
			catch (axis::foundation::OutOfBoundsException)
			{
				// test ok
			}
		}
		TEST_METHOD(IncrementRowVectorTest)
		{
			RowVector m(2);
			m.SetElement(0, 1);
			m.SetElement(1, 9);

			RowVector x(2, 2);
			x.SetElement(0, 0);
			x.SetElement(1, 3);

			// we must ensure that the same RowVector is being returned by
			// this statement
			Assert::AreSame(m, (m += x));

			Assert::AreEqual(1,  m(0), REAL_TOLERANCE);
			Assert::AreEqual(12, m(1), REAL_TOLERANCE);

			Assert::AreEqual(0, x(0), REAL_TOLERANCE);
			Assert::AreEqual(3, x(1), REAL_TOLERANCE);
		}
		TEST_METHOD(DecrementRowVectorTest)
		{
			RowVector m(2);
			m.SetElement(0, 4);
			m.SetElement(1, 13);

			RowVector x(2, 2);
			x.SetElement(0, 7);
			x.SetElement(1, 5);

			// we must ensure that the same RowVector is being returned by
			// this statement
			Assert::AreSame(m, (m -= x));

			Assert::AreEqual(-3, m(0), REAL_TOLERANCE);
			Assert::AreEqual(8,  m(1), REAL_TOLERANCE);

			Assert::AreEqual(7, x(0), REAL_TOLERANCE);
			Assert::AreEqual(5, x(1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarMultiplyOperatorTest)
		{
			RowVector m(2);
			m.SetElement(0, 4);
			m.SetElement(1, 3);
			Assert::AreEqual(4, m(0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1), REAL_TOLERANCE);

			// we must ensure that the same RowVector is being returned by
			// this statement
			Assert::AreSame(m, (m *= 10));

			Assert::AreEqual(40, m(0), REAL_TOLERANCE);
			Assert::AreEqual(30, m(1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScaleMethodTest)
		{
			RowVector m(2);
			m.SetElement(0, 4);
			m.SetElement(1, 3);
			Assert::AreEqual(4, m(0), REAL_TOLERANCE);
			Assert::AreEqual(3, m(1), REAL_TOLERANCE);

			// we must ensure that the same RowVector is being returned by
			// this statement
			Assert::AreSame(m, (RowVector&)m.Scale(10));

			Assert::AreEqual(40, m(0), REAL_TOLERANCE);
			Assert::AreEqual(30, m(1), REAL_TOLERANCE);
		}
		TEST_METHOD(ScalarDivideOperatorTest)
		{
			RowVector m(2);
			m.SetElement(0, 8);
			m.SetElement(1, 0);
			Assert::AreEqual(8, m(0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1), REAL_TOLERANCE);

			// we must ensure that the same RowVector is being returned by
			// this statement
			Assert::AreSame(m, (m /= 4));

			Assert::AreEqual(2, m(0), REAL_TOLERANCE);
			Assert::AreEqual(0, m(1), REAL_TOLERANCE);
		}
		TEST_METHOD(SelfScalarProductMethodTest)
		{
			RowVector m(4);
			m(0) = 5; m(1) = 3; m(2) = 4; m(3) = 1; 
			Assert::AreEqual(25 + 9 + 16 + 1, m.SelfScalarProduct(), REAL_TOLERANCE);
		}
		TEST_METHOD(NormMethodTest)
		{
			RowVector m(4);
			m(0) = 5; m(1) = 3; m(2) = 4; m(3) = 1; 
			real p = 25 + 9 + 16 + 1;
			Assert::AreEqual(sqrt(p), m.Norm(), REAL_TOLERANCE);
		}
		TEST_METHOD(InvertMethodTest)
		{
			ColumnVector v(3);
			v(0) = 2; v(1) = 5; v(2) = 3;
			v.Invert();

			Assert::AreEqual((real)(1.0/2.0), v(0), REAL_TOLERANCE);
			Assert::AreEqual((real)(1.0/5.0), v(1), REAL_TOLERANCE);
			Assert::AreEqual((real)(1.0/3.0), v(2), REAL_TOLERANCE);
		}
		TEST_METHOD(CopyFromTest)
		{
			RowVector v(3), r(3);
			v(0) = 3; v(1) = 7; v(2) = -8;

			r.CopyFrom(v);

			Assert::AreEqual( 3, r(0), REAL_TOLERANCE);
			Assert::AreEqual( 7, r(1), REAL_TOLERANCE);
			Assert::AreEqual(-8, r(2), REAL_TOLERANCE);
		}
    TEST_METHOD(AllocateFromModelMemoryTest)
    {
      axis::foundation::memory::RelativePointer ptr1 = 
        RowVector::Create(3);
      axis::foundation::memory::RelativePointer ptr2 = 
        RowVector::Create(3);

      RowVector& v1 = *(RowVector *)*ptr1;
      RowVector& v2 = *(RowVector *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &v1 == &v2);

      v1(0) = 0; v1(1) = -2; v1(2) = 40;
      v2(0) = 5; v2(1) = -7; v2(2) = 12;

      v1 += v2;
      Assert::AreEqual( 5, v1(0), REAL_TOLERANCE);
      Assert::AreEqual(-9, v1(1), REAL_TOLERANCE);
      Assert::AreEqual(52, v1(2), REAL_TOLERANCE);
      Assert::AreEqual( 5, v2(0), REAL_TOLERANCE);
      Assert::AreEqual(-7, v2(1), REAL_TOLERANCE);
      Assert::AreEqual(12, v2(2), REAL_TOLERANCE);

      v1.Destroy();
      v2.Destroy();
      axis::System::ModelMemory().Deallocate(ptr1);
      axis::System::ModelMemory().Deallocate(ptr2);
    }
    TEST_METHOD(AllocateFromGlobalMemoryTest)
    {
      axis::foundation::memory::RelativePointer ptr1 = 
        RowVector::CreateFromGlobalMemory(3);
      axis::foundation::memory::RelativePointer ptr2 = 
        RowVector::CreateFromGlobalMemory(3);

      RowVector& v1 = *(RowVector *)*ptr1;
      RowVector& v2 = *(RowVector *)*ptr2;

      // objects are distinct
      Assert::AreEqual(false, &v1 == &v2);

      v1(0) = 0; v1(1) = -2; v1(2) = 40;
      v2(0) = 5; v2(1) = -7; v2(2) = 12;

      v1 += v2;
      Assert::AreEqual( 5, v1(0), REAL_TOLERANCE);
      Assert::AreEqual(-9, v1(1), REAL_TOLERANCE);
      Assert::AreEqual(52, v1(2), REAL_TOLERANCE);
      Assert::AreEqual( 5, v2(0), REAL_TOLERANCE);
      Assert::AreEqual(-7, v2(1), REAL_TOLERANCE);
      Assert::AreEqual(12, v2(2), REAL_TOLERANCE);

      v1.Destroy();
      v2.Destroy();
      axis::System::GlobalMemory().Deallocate(ptr1);
      axis::System::GlobalMemory().Deallocate(ptr2);
    }
	};
}


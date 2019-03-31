#include "stdafx.h"

#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/SymmetricMatrix.hpp"
#include "foundation/blas/TriangularMatrix.hpp"
#include "foundation/blas/blas.hpp"
#include "foundation/ArgumentException.hpp"
#include "System.hpp"

namespace afb = axis::foundation::blas;

namespace axis_blas_unit_tests
{
	TEST_CLASS(LinearAlgebraTest)
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
		TEST_METHOD(TestIdentity3D)
		{
			afb::DenseMatrix eye(3,3);
      afb::Identity3D(eye);

			Assert::AreEqual(1, eye(0,0), REAL_TOLERANCE);
			Assert::AreEqual(1, eye(1,1), REAL_TOLERANCE);
			Assert::AreEqual(1, eye(2,2), REAL_TOLERANCE);
			Assert::AreEqual(0, eye(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, eye(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, eye(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, eye(1,2), REAL_TOLERANCE);
			Assert::AreEqual(0, eye(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0, eye(2,1), REAL_TOLERANCE);
		}

		TEST_METHOD(TestIdentity)
		{
			afb::SymmetricMatrix m1(3);
			afb::Identity(m1);

			Assert::AreEqual(1, m1(0,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m1(1,1), REAL_TOLERANCE);
			Assert::AreEqual(1, m1(2,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m1(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m1(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m1(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m1(1,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m1(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m1(2,1), REAL_TOLERANCE);

			afb::LowerTriangularMatrix m2(3);
			afb::Identity(m2);

			Assert::AreEqual(1, m2.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m2.GetElement(1,1), REAL_TOLERANCE);
			Assert::AreEqual(1, m2.GetElement(2,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m2.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m2.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m2.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m2.GetElement(1,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m2.GetElement(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m2.GetElement(2,1), REAL_TOLERANCE);

			afb::UpperTriangularMatrix m3(3);
			afb::Identity(m3);

			Assert::AreEqual(1, m3.GetElement(0,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m3.GetElement(1,1), REAL_TOLERANCE);
			Assert::AreEqual(1, m3.GetElement(2,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m3.GetElement(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m3.GetElement(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m3.GetElement(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m3.GetElement(1,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m3.GetElement(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m3.GetElement(2,1), REAL_TOLERANCE);

			afb::DenseMatrix m4(3,3);
			afb::Identity(m4);

			Assert::AreEqual(1, m4(0,0), REAL_TOLERANCE);
			Assert::AreEqual(1, m4(1,1), REAL_TOLERANCE);
			Assert::AreEqual(1, m4(2,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m4(0,1), REAL_TOLERANCE);
			Assert::AreEqual(0, m4(0,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m4(1,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m4(1,2), REAL_TOLERANCE);
			Assert::AreEqual(0, m4(2,0), REAL_TOLERANCE);
			Assert::AreEqual(0, m4(2,1), REAL_TOLERANCE);
		}

		TEST_METHOD(TestIdentity_ErrorIncompatible)
		{
			afb::DenseMatrix m(3,4);
			try
			{
				afb::Identity(m);
				Assert::Fail(_T("A not square matrix was accepted."));
			}
			catch (axis::foundation::ArgumentException&)
			{
				// test ok
			}
			catch (...)
			{
				Assert::Fail(_T("Unknown exception thrown."));
			}
		}

		TEST_METHOD(TestKroneckerDelta)
		{
			Assert::AreEqual((real)1.0, afb::KroneckerDelta(0,0));
			Assert::AreEqual((real)1.0, afb::KroneckerDelta(7,7));
			Assert::AreEqual((real)0.0, afb::KroneckerDelta(0,3));
			Assert::AreEqual((real)0.0, afb::KroneckerDelta(3,4));
		}

	};
}


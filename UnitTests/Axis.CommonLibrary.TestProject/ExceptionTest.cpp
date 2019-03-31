#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"
#include "AxisString.hpp"
#include "System.hpp"
#include "foundation/AxisException.hpp"
#include <set>
#include "foundation/SymbolRedefinedException.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace axis::foundation;
using namespace axis;

// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const String& s)
{
	return std::wstring(s.data());
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const AxisException& ex)
{
	return std::wstring((ex.GetTypeName() + _T(": ") + ex.GetMessage()).data());
}

namespace axis_common_library_unit_tests
{		
	TEST_CLASS(ExceptionTest)
	{
	public:
		// Class used just to test the behavior of STL
		class TestClass
		{
			std::set<SourceTraceHint> myMap;		
			String description;
		public:
			TestClass(void)
			{

			}

			TestClass(const String& desc)
			{
				description = desc;
			}
		};
		
    TEST_METHOD_INITIALIZE(SetUp)
    {
      axis::System::Initialize();
    }

    TEST_METHOD_CLEANUP(TearDown)
    {
      axis::System::Finalize();
    }

		TEST_METHOD(TestSTLMapCreation)
		{
			TestClass c1(_T("Desc"));
			TestClass c2 = c1;
		}

		TEST_METHOD(TestConstructor)
		{	// test is successful if we can run everything
			AxisException ex(_T("Meu teste"));
			AxisException ex2;
			AxisException ex3(&ex);		
		}	// destructor is called here

		TEST_METHOD(TestCloneMethod)
		{
			AxisException ex1(_T("Test1"));
			SymbolRedefinedException ex2(_T("Ex2"), String(_T("string-teste")), 23, 77);

			// clone a simple, plain exception
			AxisException& clone1 = (AxisException&)ex1.Clone();
			Assert::AreEqual(clone1.GetMessage(), ex1.GetMessage());

			// clone a more "sophisticated" exception :P
			SymbolRedefinedException& clone2 = (SymbolRedefinedException&)ex2.Clone();
			Assert::AreEqual(clone2.GetMessage(), ex2.GetMessage().data());
			Assert::AreEqual(clone2.GetColumnIndex(), ex2.GetColumnIndex());
			Assert::AreEqual(clone2.GetFileName(), ex2.GetFileName().data());
			Assert::AreEqual(clone2.GetLineIndex(), ex2.GetLineIndex());
		}

		TEST_METHOD(TestExceptionNesting)
		{
			AxisException ex1(_T("Ex1"));
			AxisException ex2(_T("Ex2"));
			AxisException ex3(_T("Ex3"));

			// nest one level
			ex2 << ex3;
			Assert::IsNotNull(ex2.GetInnerException());
			Assert::AreNotSame(ex3, *ex2.GetInnerException());	// check if it is a completely different object (a clone, that is)
			Assert::AreEqual(_T("Ex3"), ex2.GetInnerException()->GetMessage());	// check if indeed it is a clone (has the same data)

			// nest another level; at this point, ex1 contains a clone of ex2, which contains a clone of ex3
			ex1 << ex2;
			Assert::IsNotNull(ex1.GetInnerException());
			Assert::AreNotSame(ex2, *ex1.GetInnerException());	// check if it is a completely different object (a clone, that is)
			Assert::AreNotSame(ex3, *ex1.GetInnerException());
			Assert::AreEqual(_T("Ex2"), ex1.GetInnerException()->GetMessage());	// check if indeed it is a clone (has the same data)

			AxisException& e = *ex1.GetInnerException();
			Assert::IsNotNull(e.GetInnerException());	// check if it is a completely different object (a clone, that is)
			Assert::AreNotSame(ex1, *e.GetInnerException());
			Assert::AreNotSame(ex2, *e.GetInnerException());
			Assert::AreNotSame(ex3, *e.GetInnerException());
			Assert::AreEqual(_T("Ex3"), e.GetInnerException()->GetMessage());		// check if indeed it is a clone (has the same data)
		}

		TEST_METHOD(TestExceptionStackNavigation)
		{
			try
			{
				AxisException ex(_T("Meu teste"));
				AxisException ex2;
				AxisException ex3(&ex);

				throw(ex);
			}
			catch (AxisException e)
			{
				Logger::WriteMessage("Exception 1 thrown has the following message within: ");
				Logger::WriteMessage(e.GetMessage().data());
				Assert::AreEqual(_T("Meu teste"), e.GetMessage());
			}
			try
			{
				AxisException ex(_T("Meu teste"));
				AxisException ex2;
				AxisException ex3(&ex);

				throw(ex3);
			}
			catch (AxisException e)
			{
				Assert::IsNotNull(e.GetInnerException());
				Logger::WriteMessage("Exception 2 thrown has the following message within: ");
				Logger::WriteMessage(e.GetInnerException()->GetMessage().data());
				Assert::AreEqual(_T("Meu teste"), e.GetInnerException()->GetMessage());
			}
		}

		TEST_METHOD(TestTagAdditionAndQuery)
		{
			try
			{
				AxisException ex;
				ex << SourceTraceHint(1) << SourceTraceHint(4);
			}
			catch (AxisException ex)
			{
				Assert::AreEqual(true, ex.HasSourceTraceHint(SourceTraceHint(1)));
				Assert::AreEqual(true, ex.HasSourceTraceHint(SourceTraceHint(4)));
			}
		}

	};
}
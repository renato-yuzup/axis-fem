#pragma once
#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"
#include "AxisString.hpp"
#include "System.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

using namespace axis;
using namespace axis::foundation;


// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const String& s)
{
	return std::wstring(s.data());
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const String::iterator& it)
{
	return std::wstring(1, *it);
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const String::reverse_iterator& it)
{
	return std::wstring(1, *it);
}


namespace axis_common_library_unit_tests
{
	// This fixture tests behaviors of exceptions as established in axis Common Library component.
	TEST_CLASS(StringTestFixture)
	{
	public:
    TEST_METHOD_INITIALIZE(SetUp)
    {
      axis::System::Initialize();
    }

    TEST_METHOD_CLEANUP(TearDown)
    {
      axis::System::Finalize();
    }

		TEST_METHOD(TestCreation)
		{
			String testString;	// empty string
			String myString(_T("test1234"));
			String copiedString(myString);
			String partialString(myString, 4, 3);

			Assert::AreEqual(true, testString.empty());
			Assert::AreEqual(_T("test1234"), myString);
			Assert::AreEqual(_T("test1234"), copiedString);
			Assert::AreEqual(copiedString, myString);
			Assert::AreEqual(false, myString.data() == copiedString.data());
			Assert::AreEqual(_T("123"), partialString);
		}

		TEST_METHOD(TestClear)
		{
			String emptyString;
			String myString(_T("1234"));

			Assert::AreEqual(_T("1234"), myString);
			Assert::AreEqual(_T(""), emptyString);
			Assert::AreEqual((size_t)4, myString.size());
			Assert::AreEqual((size_t)0, emptyString.size());

			emptyString.clear();
			myString.clear();

			Assert::AreEqual(true, emptyString.empty());
			Assert::AreEqual(true, myString.empty());
			Assert::AreEqual(_T(""), emptyString);
			Assert::AreEqual(_T(""), emptyString);
			Assert::AreEqual((size_t)0, myString.size());
			Assert::AreEqual((size_t)0, emptyString.size());
		}

		TEST_METHOD(TestAttributes)
		{
			String blankString;
			String nonEmptyString(_T("test432"));

			Assert::AreEqual(true, blankString.empty());
			Assert::AreEqual(false, nonEmptyString.empty());

			Assert::AreEqual((size_t)0, blankString.size());
			Assert::AreEqual((size_t)7, nonEmptyString.size());

			Assert::AreEqual((size_t)0, blankString.length());
			Assert::AreEqual((size_t)7, nonEmptyString.length());

			Assert::AreEqual(true, blankString.capacity() > 0);
			Assert::AreEqual(true, nonEmptyString.capacity() > 7);
		}

		TEST_METHOD(TestAssignment)
		{
			String newString = _T("hello");
			String myString(_T("test"));

			Assert::AreEqual(_T("hello"), newString);
			Assert::AreEqual(_T("test"), myString);

			myString = newString;

			Assert::AreEqual(_T("hello"), newString);
			Assert::AreEqual(_T("hello"), myString);
		}

		TEST_METHOD(TestForwardIterators)
		{
			String bigString = _T("This is a little big string!");

			String::iterator begin = bigString.begin();
			String::iterator end = bigString.end();

			// move end iterator backward and check
			--end;
			Assert::AreEqual(_T('T'), *begin);
			Assert::AreEqual(_T('!'), *end);

			// move iterator a little and see if we can get chars of our string
			++begin;
			Assert::AreEqual(_T('h'), *begin);
			++begin;
			Assert::AreEqual(_T('i'), *begin);
			++begin;
			Assert::AreEqual(_T('s'), *begin);
			++begin;
			Assert::AreEqual(_T(' '), *begin);
			++begin;
			Assert::AreEqual(_T('i'), *begin);
			++begin;
			Assert::AreEqual(_T('s'), *begin);
			--end;
			Assert::AreEqual(_T('g'), *end);
			--end;
			Assert::AreEqual(_T('n'), *end);
			--end;
			Assert::AreEqual(_T('i'), *end);

			// check assignment and equality
			begin = bigString.begin();
			Assert::AreEqual(begin, bigString.begin());
			Assert::AreEqual(true, begin != bigString.end());

			// check large increments
			begin += 6;
			Assert::AreEqual(_T('s'), *begin);
			String::iterator testIt = begin + 5;
			Assert::AreEqual(_T('i'), *testIt);
			Assert::AreEqual(_T('s'), *begin);
			begin -= 6;
			Assert::AreEqual(_T('T'), *begin);

			// very large increments should be caught
			try
			{
				begin += 1000;
				Assert::Fail(_T("Expected exception was not thrown!"));
			}
			catch (OutOfBoundsException&)
			{
				Assert::AreEqual(_T('T'), *begin);
			}
		}

		TEST_METHOD(TestReverseIterators)
		{
			String bigString = _T("A very big string! Or not?");

			String::reverse_iterator begin = bigString.rbegin();
			String::reverse_iterator end = bigString.rend();

			// move end iterator backward and check
			--end;
			Assert::AreEqual(_T('?'), *begin);
			Assert::AreEqual(_T('A'), *end);

			// move iterator a little and see if we can get chars of our string
			++begin;
			Assert::AreEqual(_T('t'), *begin);
			++begin;
			Assert::AreEqual(_T('o'), *begin);
			++begin;
			Assert::AreEqual(_T('n'), *begin);
			++begin;
			Assert::AreEqual(_T(' '), *begin);
			++begin;
			Assert::AreEqual(_T('r'), *begin);
			++begin;
			Assert::AreEqual(_T('O'), *begin);
			--end;
			Assert::AreEqual(_T(' '), *end);
			--end;
			Assert::AreEqual(_T('v'), *end);
			--end;
			Assert::AreEqual(_T('e'), *end);

			// check assignment and equality
			begin = bigString.rbegin();
			Assert::AreEqual(begin, bigString.rbegin());
			Assert::AreEqual(true, begin != bigString.rend());

			// check large increments
			begin += 6;
			Assert::AreEqual(_T('O'), *begin);
			String::reverse_iterator testIt = begin + 5;
			Assert::AreEqual(_T('i'), *testIt);
			Assert::AreEqual(_T('O'), *begin);
			begin -= 6;
			Assert::AreEqual(_T('?'), *begin);

			// very large increments should be caught
			try
			{
				begin += 1000;
				Assert::Fail(_T("Expected exception was not thrown!"));
			}
			catch (OutOfBoundsException&)
			{
				Assert::AreEqual(_T('?'), *begin);
			}
		}

		TEST_METHOD(TestAppend)
		{
			String test(_T("test"));
			String hello(_T("hello"));
			String helloTest;

			Assert::AreEqual((size_t)0, helloTest.size());
			helloTest.append(hello);
			helloTest += _T(' ');
			helloTest.append(test);
			helloTest.append(_T("!!!"));

			helloTest.push_back(_T('!'));

			Assert::AreEqual(_T("hello test!!!!"), helloTest);
		}

		TEST_METHOD(TestInsert)
		{
			// first, let's try to insert at the beginning and the end
			String test(_T("test"));
			String hello(_T("hello"));
			String exclamation(_T("!!!"));
			String helloTest;

			Assert::AreEqual((size_t)0, helloTest.size());
			helloTest.insert(helloTest.end(), test.begin(), test.end());
			helloTest.insert(helloTest.begin(), hello.begin(), hello.end());

			helloTest.insert(5, 1, _T(' '));

			helloTest.insert(helloTest.end(), exclamation.begin(), exclamation.end());

			Assert::AreEqual(_T("hello test!!!"), helloTest);

			String testStr;
			testStr.insert(testStr.end(), _T('a'));
			Assert::AreEqual(_T("a"), testStr);
			testStr.insert(testStr.end(), _T('b'));
			Assert::AreEqual(_T("ab"), testStr);
			testStr.insert(testStr.end(), _T('c'));
			Assert::AreEqual(_T("abc"), testStr);
			testStr.insert(testStr.end(), _T('d'));
			Assert::AreEqual(_T("abcd"), testStr);
		}

		TEST_METHOD(TestErase)
		{
			String testString = _T("We will erase some words here");

			testString.erase(14, 5);
			Assert::AreEqual(_T("We will erase words here"), testString);
			String::iterator it = testString.begin() + 19;
			testString.erase(it);
			Assert::AreEqual(_T("We will erase words"), testString);
		}

		TEST_METHOD(TestTrim)
		{
			String testLeft = _T("   \t\n\t\n   Trim here! ");
			String testRight = _T("   Trim here!  \n\n\t    \t  ");
			String testBoth = _T("  \t\n   \r   Trim both!\t    \t\n\r  ");
			String testNone = _T("c  Can't trim    \t any");

			Assert::AreEqual(_T("Trim here! "), testLeft.trim_left());
			Assert::AreEqual(_T("   Trim here!"), testRight.trim_right());
			Assert::AreEqual(_T("Trim both!"), testBoth.trim());
			Assert::AreEqual(_T("c  Can't trim    \t any"), testNone.trim());
		}

		TEST_METHOD(TestReplace)
		{
			String testString = _T("We need a test!");
			testString.replace(3, 4, _T("won't do"));

			Assert::AreEqual(_T("We won't do a test!"), testString);
		}

		TEST_METHOD(TestSubstring)
		{
			String testString = _T("Check this out!");
			String substr = testString.substr(6, 4);

			Assert::AreEqual(_T("this"), substr);
		}

		TEST_METHOD(TestReverse)
		{
			String s = _T("To be reversed");

			s.reverse();
			Assert::AreEqual(_T("desrever eb oT"), s);

			s.reverse();
			Assert::AreEqual(_T("To be reversed"), s);
		}

		TEST_METHOD(TestFindFirst)
		{
			String s = _T("Must find this pattern!");
			String pattern = _T("this");
			Assert::AreEqual((size_t)10, s.find(pattern));
		}

		TEST_METHOD(TestFindLast)
		{
			String s = _T("Must find in this pattern!");
			String pattern = _T("t");
			Assert::AreEqual((size_t)21, s.rfind(pattern));
		}

		TEST_METHOD(TestFindFirstOf)
		{
			String s = _T("Must find in this pattern!");
			String pattern = _T("fid");
			Assert::AreEqual((size_t)5, s.find_first_of(pattern));
		}

		TEST_METHOD(TestFindLastOf)
		{
			String s = _T("Must find in this pattern!");
			String pattern = _T("fid");
			Assert::AreEqual((size_t)15, s.find_last_of(pattern));
		}

		TEST_METHOD(TestFindFirstNotOf)
		{
			String abc = _T("abc123");
			Assert::AreEqual((size_t)2, abc.find_first_not_of(_T("ab")));
		}

		TEST_METHOD(TestFindLastNotOf)
		{
			String s = _T("Must find in this pattern!");
			String pattern = _T("fidn!t");
			Assert::AreEqual((size_t)23, s.find_last_not_of(pattern));
		}
	};
}



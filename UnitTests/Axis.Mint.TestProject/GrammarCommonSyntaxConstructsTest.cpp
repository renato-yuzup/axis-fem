#pragma once
#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"
#include "services/language/syntax/BlockHeaderParser.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/syntax/BlockTailParser.hpp"
#include "System.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

using namespace axis::services::language::syntax;
using namespace axis::services::language::syntax::evaluation;
using namespace axis::services::language::iterators;
using namespace axis::services::language::factories;
using namespace axis::services::language::parsing;
using namespace axis;


// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const String& s)
{
	return std::wstring(s.data());
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const InputIterator& it)
{
	return std::wstring(_T("Iterator at '")) + *it + _T("'");
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const ParseResult::Result& result)
{
	return std::wstring(String::int_parse((long)result).data());
}

namespace axis_common_library_unit_tests
{
	// This fixture tests behaviors of exceptions as established in axis Common Library component.
	TEST_CLASS(SyntaxConstructsParserTestFixture)
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

		TEST_METHOD(TestSimpleBlockDeclarationFullMatch)
		{
			BlockHeaderParser parser;
			String blockName = _T("myBlock_Id");
			String expression = _T("BEGIN ");
			expression.append(blockName);

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(false, parser.HasParameters());
			Assert::AreEqual(blockName, parser.GetBlockName());
		}

		TEST_METHOD(TestSimpleBlockDeclarationPartialMatch)
		{
			BlockHeaderParser parser;
			String expression = _T("BEGIN ");

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
			Assert::AreEqual(false, parser.HasParameters());
			Assert::AreEqual(String(), parser.GetBlockName());
		}

		TEST_METHOD(TestSimpleBlockDeclarationFailMatch)
		{
			BlockHeaderParser parser;
			String blockName = _T("myBlock_Id");
			String expression = _T("BEGIN");
			expression.append(blockName);

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(false, parser.HasParameters());
			Assert::AreEqual(String(), parser.GetBlockName());
		}

		TEST_METHOD(TestComplexBlockDeclarationFullMatch)
		{
			BlockHeaderParser parser;
			String blockName = _T("myBlock_Id");
			String expression = _T("BEGIN ");
			String params = _T("param1 = test, param2 = 1, param3 = (arg1 = 0, arg2 = a, 0.4, (0, 1, 2, 3, () ), arg3)");
			expression.append(blockName).append(_T(" WITH ")).append(params);

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(true, parser.HasParameters());
			Assert::AreEqual(end, result.GetLastReadPosition());
			Assert::AreEqual(blockName, parser.GetBlockName());
			const ParameterList& paramList = parser.GetParameterList();
			Assert::AreEqual(3, paramList.Count());	
			ParameterValue& val = *(++++paramList.begin())->Value;
			Assert::AreEqual(true, val.IsArray());
		}

		TEST_METHOD(TestComplexBlockDeclarationPartialMatch)
		{
			BlockHeaderParser parser;
			String blockName = _T("myBlock_Id");
			String expression = _T("BEGIN ");
			String params = _T("param1 = test, param2 = 1, param3 = ");
			expression.append(blockName).append(_T(" WITH ")).append(params);

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
			Assert::AreEqual(false, parser.HasParameters());
			Assert::AreEqual(String(), parser.GetBlockName());
		}

		TEST_METHOD(TestComplexBlockDeclarationFailMatch)
		{
			BlockHeaderParser parser;
			String blockName = _T("myBlock_Id");
			String expression = _T("BEGIN ");
			String params = _T("param1 = test, param2 = 1, 0");
			expression.append(blockName).append(_T(" WITH ")).append(params);

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(false, parser.HasParameters());
			Assert::AreEqual(String(), parser.GetBlockName());
		}

		TEST_METHOD(TestBlockTailDeclarationFullMatch)
		{
			BlockTailParser parser;
			String blockName = _T("myBlock_Id");
			String expression = _T("END \t ");
			expression.append(blockName).append(_T("\t\t "));

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(blockName, parser.GetBlockName());
			Assert::AreEqual(end, result.GetLastReadPosition());
		}

		TEST_METHOD(TestBlockTailDeclarationPartialMatch)
		{
			BlockTailParser parser;
			String expression = _T("END \t ");

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
			Assert::AreEqual(String(_T("")), parser.GetBlockName());
			Assert::AreEqual(end, result.GetLastReadPosition());
		}

		TEST_METHOD(TestBlockTailDeclarationFailMatch)
		{
			BlockTailParser parser;
			String blockName = _T("myBlock_Id");
			String expression = _T("END \t +");
			expression.append(blockName).append(_T("\t\t "));

			InputIterator begin = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			ParseResult result = parser(begin, end);

			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(String(_T("")), parser.GetBlockName());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
		}
	};
}




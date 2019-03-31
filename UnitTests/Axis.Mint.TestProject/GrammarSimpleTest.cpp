#pragma once
#include "StdAfx.h"
#include "services/language/primitives/Parser.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "AxisString.hpp"
#include "System.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/IdTerminal.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "services/language/parsing/StringTerminal.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "services/language/parsing/ReservedWordTerminal.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

using namespace axis::services::language::primitives;
using namespace axis::services::language::iterators;
using namespace axis::services::language::factories;
using namespace axis::services::language::parsing;
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
	TEST_CLASS(SimpleGrammarTestFixture)
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

		TEST_METHOD(TestIdParserFullMatch)
		{
			Parser parser = AxisGrammar::CreateIdParser();
			String testString = _T("_This_is_an_id0");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Logger::WriteMessage(node.ToString());
			Assert::AreEqual(testString, node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(true, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(false, term.IsString());

			IdTerminal& idTerm = (IdTerminal&)term;
			Assert::AreEqual(testString, idTerm.GetId());
		}

		TEST_METHOD(TestIdParserPartialMatch)
		{
			Parser parser = AxisGrammar::CreateIdParser();
			String testString = _T("_This_is_an_id0 and here we have more to parse");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end, false);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());

			// ensure we didn't run into the end of the string
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(axis::String(_T(" and here we have more to parse")), result.GetLastReadPosition().ToString(end));
			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(axis::String(_T("_This_is_an_id0")), node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(true, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(false, term.IsString());
		}

		TEST_METHOD(TestIdParserFailMatch)
		{
			Parser parser = AxisGrammar::CreateIdParser();
			String testString = _T("0FailedId!");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());

			// ensure we failed
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(result.GetLastReadPosition(), IteratorFactory::CreateStringIterator(testString));		
			Assert::AreEqual(testString, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestNumberParserFullMatch)
		{
			{
				/* FIRST PASS: CHECK FOR FLOATING-POINT NUMBERS */
				Parser doubleParser = AxisGrammar::CreateNumberParser();
				String testDoubleString = _T("+45.59e-2");
				InputIterator it = IteratorFactory::CreateStringIterator(testDoubleString);
				InputIterator end = IteratorFactory::CreateStringIterator(testDoubleString.end());
				ParseResult result = doubleParser.Parse(it, end);

				// validate results
				Assert::AreEqual(true, result.IsMatch());
				Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
				Assert::AreEqual(result.GetLastReadPosition(), end);

				// check that we didn't change the client operators
				Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testDoubleString));		

				// check syntax tree
				ParseTreeNode& node = result.GetParseTree();
				Assert::AreEqual(true, node.IsRoot());
				Assert::AreEqual(true, node.IsTerminal());
				Assert::AreEqual(true, node.GetNextSibling() == NULL);
				Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
				Assert::AreEqual(true, node.GetParent() == NULL);
				Assert::AreEqual(testDoubleString, node.ToString());

				// since it is a terminal, this cast is valid
				SymbolTerminal& term = (SymbolTerminal&)node;
				Assert::AreEqual(false, term.IsId());
				Assert::AreEqual(true, term.IsNumber());
				Assert::AreEqual(false, term.IsOperator());
				Assert::AreEqual(false, term.IsReservedWord());
				Assert::AreEqual(false, term.IsString());

				NumberTerminal& doubleTerm = (NumberTerminal&)term;
				Assert::AreEqual(false, doubleTerm.IsInteger());
				Assert::AreEqual(true, abs(doubleTerm.GetDouble() - 45.59e-2) <= 1e-8);	/* that is, abs(...) <= 1e-8? */
				Assert::AreEqual(0L, doubleTerm.GetInteger());

			}

			{
				/* SECOND PASS: CHECK FOR INTEGER NUMBERS */
				Parser intParser = AxisGrammar::CreateNumberParser();
				String testIntString = _T("-289455");
				InputIterator it = IteratorFactory::CreateStringIterator(testIntString);
				InputIterator end = IteratorFactory::CreateStringIterator(testIntString.end());
				ParseResult result = intParser.Parse(it, end);

				// validate results
				Assert::AreEqual(true, result.IsMatch());
				Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
				Assert::AreEqual(result.GetLastReadPosition(), end);

				// check that we didn't change the client operators
				Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testIntString));		

				// check syntax tree
				ParseTreeNode& node = result.GetParseTree();
				Assert::AreEqual(true, node.IsRoot());
				Assert::AreEqual(true, node.IsTerminal());
				Assert::AreEqual(true, node.GetNextSibling() == NULL);
				Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
				Assert::AreEqual(true, node.GetParent() == NULL);
				Assert::AreEqual(testIntString, node.ToString());

				// since it is a terminal, this cast is valid
				SymbolTerminal& term = (SymbolTerminal&)node;
				Assert::AreEqual(false, term.IsId());
				Assert::AreEqual(true, term.IsNumber());
				Assert::AreEqual(false, term.IsOperator());
				Assert::AreEqual(false, term.IsReservedWord());
				Assert::AreEqual(false, term.IsString());

				NumberTerminal& intTerm = (NumberTerminal&)term;
				Assert::AreEqual(true, intTerm.IsInteger());
				Assert::AreEqual(true, abs(intTerm.GetDouble() - (-289455)) <= 1e-8);		/* that is, abs(...) <= 1e-8? */
				Assert::AreEqual(-289455L, intTerm.GetInteger());
			}
		}

		TEST_METHOD(TestNumberParserPartialMatch)
		{
			Parser parser = AxisGrammar::CreateNumberParser();
			String testString = _T("-2.9687e-5(and more to go)");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());

			// ensure we didn't run into the end of the string
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(axis::String(_T("(and more to go)")), result.GetLastReadPosition().ToString(end));
			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(axis::String(_T("-2.9687e-5")), node.ToString());

			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(true, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(false, term.IsString());
		}

		TEST_METHOD(TestNumberParserFailMatch)
		{
			Parser parser = AxisGrammar::CreateNumberParser();
			String testString = _T("This is not a number");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());

			// ensure we failed
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(result.GetLastReadPosition(), IteratorFactory::CreateStringIterator(testString));		
			Assert::AreEqual(testString, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestNumberParserIntOverflow)
		{
			Parser parser = AxisGrammar::CreateNumberParser();
			String overflowNum = _T("-999999999999999999999999999999");
			InputIterator it = IteratorFactory::CreateStringIterator(overflowNum);
			InputIterator end = IteratorFactory::CreateStringIterator(overflowNum.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());

			// ensure we failed
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(result.GetLastReadPosition(), IteratorFactory::CreateStringIterator(overflowNum));		
			Assert::AreEqual(overflowNum, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestNumberParserDoubleOverflow)
		{
			Parser parser = AxisGrammar::CreateNumberParser();
			String overflowNum = _T("-999999999999999.999999999999999E+9999999");
			InputIterator it = IteratorFactory::CreateStringIterator(overflowNum);
			InputIterator end = IteratorFactory::CreateStringIterator(overflowNum.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());

			// ensure we failed
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(result.GetLastReadPosition(), IteratorFactory::CreateStringIterator(overflowNum));		
			Assert::AreEqual(overflowNum, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestBlankParserFullMatch)
		{
			Parser blankParser = AxisGrammar::CreateBlankParser(false);
			String testBlankString = _T("  \t  \t\t\t   \t");
			InputIterator it = IteratorFactory::CreateStringIterator(testBlankString);
			InputIterator end = IteratorFactory::CreateStringIterator(testBlankString.end());
			ParseResult result = blankParser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testBlankString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());


			// check for no space success
			testBlankString = _T("No space at the beginning");
			InputIterator begin = IteratorFactory::CreateStringIterator(testBlankString);
			end = IteratorFactory::CreateStringIterator(testBlankString.end());
			result = blankParser.Parse(begin, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), begin);
		}

		TEST_METHOD(TestBlankParserPartialMatch)
		{
			Parser blankParser = AxisGrammar::CreateBlankParser(false);
			String testBlankString = _T("  \t  \tMore to go");
			InputIterator it = IteratorFactory::CreateStringIterator(testBlankString);
			InputIterator end = IteratorFactory::CreateStringIterator(testBlankString.end());
			ParseResult result = blankParser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(axis::String(_T("More to go")), result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestBlankParserFailMatch)
		{
			Parser blankParser = AxisGrammar::CreateBlankParser(true);
			String testBlankString = _T("No spaces at the beginning");
			InputIterator begin = IteratorFactory::CreateStringIterator(testBlankString);
			InputIterator end = IteratorFactory::CreateStringIterator(testBlankString.end());
			ParseResult result = blankParser.Parse(begin, end);

			// validate results
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), begin);		
			Assert::AreEqual(axis::String(_T("No spaces at the beginning")), result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestStringNoEscapeParserFullMatch)
		{
			Parser parser = AxisGrammar::CreateStringParser();
			String stringContents = _T("A string \\\\ without escape chars");
			String testString = _T("\"A string \\\\ without escape chars\"");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(stringContents, node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(true, term.IsString());
		}

		TEST_METHOD(TestStringNoEscapeParserPartialMatch)
		{
			Parser parser = AxisGrammar::CreateStringParser();
			String stringContents = _T("My String");
			String testString = _T("\"My String\"+more to go");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());

			// ensure we didn't run into the end of the string
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(axis::String(_T("+more to go")), result.GetLastReadPosition().ToString(end));
			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(stringContents, node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(true, term.IsString());
		}

		TEST_METHOD(TestStringNoEscapeParserFailMatch)
		{
			Parser parser = AxisGrammar::CreateStringParser();
			String testString = _T("\"My String without closing quote");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());

			// ensure we didn't run into the end of the string
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(testString, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestStringWithEscapeParserFullMatch)
		{
			Parser parser = AxisGrammar::CreateStringParser(true);
			String stringContents = _T("A string \" with some \\ escape chars");
			String testString = _T("\"A string \\\" with some \\\\ escape chars\"");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(stringContents, node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(true, term.IsString());
		}

		TEST_METHOD(TestStringWithEscapeParserPartialMatch)
		{
			Parser parser = AxisGrammar::CreateStringParser(true);
			String stringContents = _T("My \\ \"String");
			String testString = _T("\"My \\\\ \\\"String\"+more to go");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());

			// ensure we didn't run into the end of the string
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(axis::String(_T("+more to go")), result.GetLastReadPosition().ToString(end));
			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(stringContents, node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(true, term.IsString());
		}

		TEST_METHOD(TestStringWithEscapeParserFailMatch)
		{
			Parser parser = AxisGrammar::CreateStringParser(true);
			String testString = _T("\"My String with \\t invalid escape char\"");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());

			// ensure we didn't run into the end of the string
			Assert::AreNotEqual(result.GetLastReadPosition(), end);		
			Assert::AreEqual(testString, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestOperatorParserFullMatch)
		{
			Parser parser = AxisGrammar::CreateOperatorParser(_T("MY_OPERATOR"), 1);	/* dummy associated value; just for test purposes */
			String testString = _T("MY_OPERATOR");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(testString, node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(true, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(false, term.IsString());

			// get operator values from the syntax node
			OperatorTerminal& opTerm = (OperatorTerminal&)term;
			Assert::AreEqual(1, opTerm.GetValue());
		}

		TEST_METHOD(TestOperatorParserPartialMatch)
		{
			Parser parser = AxisGrammar::CreateOperatorParser(_T("MY_OPERATOR"), 1);	/* dummy associated value; just for test purposes */
			String testString = _T("MY_OPERATOR=\tand some trailing string");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreNotEqual(result.GetLastReadPosition(), end);
			Assert::AreEqual(axis::String(_T("=\tand some trailing string")), result.GetLastReadPosition().ToString(end));

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(axis::String(_T("MY_OPERATOR")), node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(true, term.IsOperator());
			Assert::AreEqual(false, term.IsReservedWord());
			Assert::AreEqual(false, term.IsString());

			// get operator values from the syntax node2
			OperatorTerminal& opTerm = (OperatorTerminal&)term;
			Assert::AreEqual(1, opTerm.GetValue());
		}

		TEST_METHOD(TestOperatorParserFailMatch)
		{
			Parser parser = AxisGrammar::CreateOperatorParser(_T("MY_OPERATOR"), 1);	/* dummy associated value; just for test purposes */
			String testString = _T("NOT_MY_OPERATOR");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreNotEqual(result.GetLastReadPosition(), end);
			Assert::AreEqual(testString, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode *node = &result.GetParseTree();
			Assert::AreEqual(true, node->IsEmpty());

			testString = _T("MY_OPERATOR123");	// this should also not accept
			it = IteratorFactory::CreateStringIterator(testString);
			end = IteratorFactory::CreateStringIterator(testString.end());
			result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreNotEqual(result.GetLastReadPosition(), end);
			Assert::AreEqual(testString, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			node = &result.GetParseTree();
			Assert::AreEqual(true, node->IsEmpty());
		}

		TEST_METHOD(TestReservedWordParserFullMatch)
		{
			Parser parser = AxisGrammar::CreateReservedWordParser(_T("MY_RESERVED_WORD"), 1);	/* dummy associated value; just for test purposes */
			String testString = _T("MY_RESERVED_WORD");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(testString, node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(true, term.IsReservedWord());
			Assert::AreEqual(false, term.IsString());

			// get operator values from the syntax node
			ReservedWordTerminal& resWordTerm = (ReservedWordTerminal&)term;
			Assert::AreEqual(1, resWordTerm.GetValue());
		}

		TEST_METHOD(TestReservedWordParserPartialMatch)
		{
			Parser parser = AxisGrammar::CreateReservedWordParser(_T("MY_RESERVED_WORD"), 1);	/* dummy associated value; just for test purposes */
			String testString = _T("MY_RESERVED_WORD\tand some trailing string");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end, false);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreNotEqual(result.GetLastReadPosition(), end);
			Assert::AreEqual(axis::String(_T("\tand some trailing string")), result.GetLastReadPosition().ToString(end));

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::AreEqual(true, node.IsTerminal());
			Assert::AreEqual(true, node.GetNextSibling() == NULL);
			Assert::AreEqual(true, node.GetPreviousSibling() == NULL);
			Assert::AreEqual(true, node.GetParent() == NULL);
			Assert::AreEqual(axis::String(_T("MY_RESERVED_WORD")), node.ToString());

			// since it is a terminal, this cast is valid
			SymbolTerminal& term = (SymbolTerminal&)node;
			Assert::AreEqual(false, term.IsId());
			Assert::AreEqual(false, term.IsNumber());
			Assert::AreEqual(false, term.IsOperator());
			Assert::AreEqual(true, term.IsReservedWord());
			Assert::AreEqual(false, term.IsString());

			// get operator values from the syntax node
			ReservedWordTerminal& opTerm = (ReservedWordTerminal&)term;
			Assert::AreEqual(1, opTerm.GetValue());
		}

		TEST_METHOD(TestReservedWordParserFailMatch)
		{
			Parser parser = AxisGrammar::CreateReservedWordParser(_T("MY_RESERVED_WORD"), 1);	/* dummy associated value; just for test purposes */
			String testString = _T("NOT_MY_RESERVED_WORD");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreNotEqual(result.GetLastReadPosition(), end);
			Assert::AreEqual(testString, result.GetLastReadPosition().ToString(end));

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsEmpty());
		}

		TEST_METHOD(TestEoiParserFullMatch)
		{
			Parser parser = AxisGrammar::CreateEoiParser();	/* dummy associated value; just for test purposes */
			String testString = _T("   \t\t\t  ");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode *node = &result.GetParseTree();
			Assert::AreEqual(true, node->IsRoot());
			Assert::AreEqual(true, node->IsEmpty());

			// check against when trimming whitespace is disabled
			testString = _T("");
			it = IteratorFactory::CreateStringIterator(testString);
			end = IteratorFactory::CreateStringIterator(testString.end());
			result = parser.Parse(it, end, false);


			// validate results
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), end);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			node = &result.GetParseTree();
			Assert::AreEqual(true, node->IsRoot());
			Assert::AreEqual(true, node->IsEmpty());
		}

		TEST_METHOD(TestEoiParserFailMatch)
		{
			Parser parser = AxisGrammar::CreateEoiParser();	/* dummy associated value; just for test purposes */
			String testString = _T("\tABC");
			InputIterator it = IteratorFactory::CreateStringIterator(testString);
			InputIterator end = IteratorFactory::CreateStringIterator(testString.end());
			InputIterator errorPos = IteratorFactory::CreateStringIterator(++testString.begin());
			ParseResult result = parser.Parse(it, end);

			// validate results
			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), errorPos);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			ParseTreeNode *node = &result.GetParseTree();
			Assert::AreEqual(true, node->IsRoot());
			Assert::AreEqual(true, node->IsEmpty());

			// check against when trimming whitespace is disabled
			testString = _T("\t");
			it = IteratorFactory::CreateStringIterator(testString);
			end = IteratorFactory::CreateStringIterator(testString.end());
			result = parser.Parse(it, end, false);


			// validate results
			Assert::AreEqual(true, !result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(result.GetLastReadPosition(), it);

			// check that we didn't change the client operators
			Assert::AreEqual(it, IteratorFactory::CreateStringIterator(testString));		

			// check syntax tree
			node = &result.GetParseTree();
			Assert::AreEqual(true, node->IsRoot());
			Assert::AreEqual(true, node->IsEmpty());
		}	
	};
}


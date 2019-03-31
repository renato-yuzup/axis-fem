#pragma once
#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"

#include "AxisString.hpp"
#include "System.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/primitives/Parser.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/AssignmentParser.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/IdTerminal.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "services/language/parsing/StringTerminal.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "services/language/parsing/ReservedWordTerminal.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/AssignmentExpression.hpp"
#include "services/language/parsing/EnumerationExpression.hpp"

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
	TEST_CLASS(ComplexGrammarTestFixture)
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

		TEST_METHOD(TestGeneralExpressionFullMatch)
		{
			String id = _T("My_Id001");
			String op = _T("=");
			String value = _T("-14e-7");
			double parsedValue = -14e-7;

			// build expression
			String expression = id;
			expression.append(op).append(value);

			// build our chained parser!
			GeneralExpressionParser expressionParser;
			expressionParser << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("=")) << AxisGrammar::CreateNumberParser();
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = expressionParser.Parse(it, end);

			// check result
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(ParseResult::MatchOk, result.GetResult());
			Assert::AreEqual(end, result.GetLastReadPosition());	// read until the end?

			// check syntax tree
			ParseTreeNode& node = result.GetParseTree();
			Assert::AreEqual(true, node.IsRoot());
			Assert::IsNull(node.GetNextSibling());
			Assert::AreEqual(false, node.IsEmpty());
			Assert::AreEqual(false, node.IsTerminal());

			ExpressionNode& expr = (ExpressionNode&)node;
			Assert::AreEqual(false, expr.IsAssignment());
			Assert::AreEqual(false, expr.IsEnumeration());
			Assert::AreEqual(true, expr.IsRhs());
			Assert::AreEqual(false, expr.IsEmpty());

			// go to children nodes and check them
			const ParseTreeNode *child = expr.GetFirstChild();
			const SymbolTerminal *term;
			Assert::AreEqual(true, child->IsTerminal());
			Assert::AreEqual(false, child->IsRoot());
			term = (const SymbolTerminal *)child;
			Assert::AreEqual(true, term->IsId());
			Assert::AreEqual(false, term->IsNumber());
			Assert::AreEqual(false, term->IsString());
			Assert::AreEqual(false, term->IsOperator());
			Assert::AreEqual(false, term->IsReservedWord());
			Assert::IsNotNull(term->GetNextSibling());

			child = child->GetNextSibling();
			Assert::AreEqual(true, child->IsTerminal());
			Assert::AreEqual(false, child->IsRoot());
			term = (const SymbolTerminal *)child;
			Assert::AreEqual(false, term->IsId());
			Assert::AreEqual(false, term->IsNumber());
			Assert::AreEqual(false, term->IsString());
			Assert::AreEqual(true, term->IsOperator());
			Assert::AreEqual(false, term->IsReservedWord());
			Assert::IsNotNull(term->GetNextSibling());

			child = child->GetNextSibling();
			Assert::AreEqual(true, child->IsTerminal());
			Assert::AreEqual(false, child->IsRoot());
			term = (const SymbolTerminal *)child;
			Assert::AreEqual(false, term->IsId());
			Assert::AreEqual(true, term->IsNumber());
			Assert::AreEqual(false, term->IsString());
			Assert::AreEqual(false, term->IsOperator());
			Assert::AreEqual(false, term->IsReservedWord());
			Assert::IsNull(term->GetNextSibling());
		}

		TEST_METHOD(TestGeneralExpressionPartialMatch)
		{
			String id = _T("Id0_variable");
			String op = _T("=");

			// build expression
			String expression = id;
			expression.append(op);

			// build our chained parser!
			GeneralExpressionParser expressionParser;
			expressionParser << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("=")) << AxisGrammar::CreateNumberParser();
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = expressionParser.Parse(it, end);

			// check if detected a partial match
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
		}

		TEST_METHOD(TestGeneralExpressionFailedMatch)
		{
			String id = _T("Id0_variable");
			String op = _T("++");

			// build expression
			String expression = id;
			expression.append(op);

			// build our chained parser!
			GeneralExpressionParser expressionParser;
			expressionParser << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("=")) << AxisGrammar::CreateNumberParser();
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = expressionParser.Parse(it, end);

			// check if failed
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
		}

		TEST_METHOD(TestAlternativeExpressionFullMatch)
		{
			// build expression
			String expression = _T("1234");

			// build our chained parser!
			OrExpressionParser expressionParser;
			expressionParser << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("1234==")) << AxisGrammar::CreateNumberParser();
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = expressionParser.Parse(it, end);

			// check if matched
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(false, result.GetParseTree().IsEmpty());
			Assert::AreEqual(true, result.GetParseTree().IsTerminal());
		}

		TEST_METHOD(TestAlternativeExpressionPartialMatch)
		{
			String id = _T("My_Id");
			String op = _T("=");
			String value = _T("0");

			// build expression
			String expression = id;
			expression.append(op);

			// build our chained parser!
			GeneralExpressionParser expressionParser;
			expressionParser << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("=")) << AxisGrammar::CreateNumberParser();
			OrExpressionParser altExpr;
			altExpr << AxisGrammar::CreateNumberParser() << expressionParser << AxisGrammar::CreateStringParser();

			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = altExpr.Parse(it, end);

			// check result
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
			Assert::AreEqual(end, result.GetLastReadPosition());	// read until the end?
		}

		TEST_METHOD(TestAlternativeExpressionFailedMatch)
		{
			String id = _T("My_Id");
			String op = _T("++");
			String value = _T("0");

			// build expression
			String expression = id;
			expression.append(op);

			// build our chained parser!
			GeneralExpressionParser expressionParser;
			expressionParser << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("=")) << AxisGrammar::CreateNumberParser();
			OrExpressionParser altExpr;
			altExpr << expressionParser << AxisGrammar::CreateStringParser();

			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			String::iterator x = expression.begin();
			x += 5;
			InputIterator errorPos = IteratorFactory::CreateStringIterator(x);

			// parse it!
			ParseResult result = altExpr.Parse(it, end);

			// check result
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
			Assert::AreNotEqual(errorPos, result.GetLastReadPosition());	// read until the end?
		}

		TEST_METHOD(TestAssignmentExpressionFullMatch)
		{
			// build expression
			String expression = _T("id\t = \t\t  1234+15");

			// build our chained parser!
			GeneralExpressionParser rhs;
			rhs << AxisGrammar::CreateNumberParser() << AxisGrammar::CreateOperatorParser(_T("+")) << AxisGrammar::CreateNumberParser();
			AssignmentParser assignment;
			assignment.SetRhsExpression(rhs);

			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = assignment.Parse(it, end);

			// check if matched
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(false, result.GetParseTree().IsEmpty());
			Assert::AreEqual(false, result.GetParseTree().IsTerminal()); // that is, an expression

			ExpressionNode& exprNode = (ExpressionNode&)result.GetParseTree();
			Assert::AreEqual(true, exprNode.IsAssignment());
			Assert::AreEqual(false, exprNode.IsEnumeration());
			Assert::AreEqual(false, exprNode.IsRhs());

			AssignmentExpression& assignNode = (AssignmentExpression&)exprNode;
			Assert::AreEqual(true, assignNode.GetLhs().IsTerminal());
			Assert::AreEqual(false, assignNode.GetRhs().IsTerminal());
		}

		TEST_METHOD(TestAssignmentExpressionPartialMatch)
		{
			// build expression
			String expression = _T("id\t = \t\t  ");

			// build our chained parser!
			GeneralExpressionParser rhs;
			rhs << AxisGrammar::CreateNumberParser() << AxisGrammar::CreateOperatorParser(_T("+")) << AxisGrammar::CreateNumberParser();
			AssignmentParser assignment;
			assignment.SetRhsExpression(rhs);

			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = assignment.Parse(it, end);

			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult()); // that is, an expression
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
			Assert::AreEqual(end, result.GetLastReadPosition());
		}

		TEST_METHOD(TestAssignmentExpressionFailedMatch)
		{
			// build expression
			String expression = _T("id\t + \t\t  ");

			// build our chained parser!
			GeneralExpressionParser rhs;
			rhs << AxisGrammar::CreateNumberParser() << AxisGrammar::CreateOperatorParser(_T("+")) << AxisGrammar::CreateNumberParser();
			AssignmentParser assignment;
			assignment.SetRhsExpression(rhs);

			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());
			axis::String::iterator x = expression.begin();
			x += 4;
			InputIterator errorPos = IteratorFactory::CreateStringIterator(x);


			// parse it!
			ParseResult result = assignment.Parse(it, end);

			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult()); // that is, an expression
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
			Assert::AreEqual(errorPos, result.GetLastReadPosition());
		}

		TEST_METHOD(TestEnumerationExpressionFullMatch)
		{
			// build expression
			String expression = _T("2.3e-5, 4.58    \t,90");

			// build our chained parser!
			EnumerationParser enumeration(AxisGrammar::CreateNumberParser());
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = enumeration.Parse(it, end);

			// check if matched
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(false, result.GetParseTree().IsEmpty());
			Assert::AreEqual(false, result.GetParseTree().IsTerminal()); // that is, an expression

			ExpressionNode& exprNode = (ExpressionNode&)result.GetParseTree();
			Assert::AreEqual(false, exprNode.IsAssignment());
			Assert::AreEqual(true, exprNode.IsEnumeration());
			Assert::AreEqual(false, exprNode.IsRhs());
			Assert::AreEqual(false, exprNode.IsEmpty());

			EnumerationExpression& enumNode = (EnumerationExpression&)exprNode;
		}

		TEST_METHOD(TestEnumerationExpressionPartialMatch)
		{
			/****** CASE 1 : PARTIAL MATCH WHEN ENUMERATION IS INCOMPLETE ******/
			String expression = _T("2.3e-5, 4.58    \t,     \t\t ");

			// build our chained parser!
			EnumerationParser enumeration(AxisGrammar::CreateNumberParser());
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = enumeration.Parse(it, end);

			// check if matched
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());


			/****** CASE 2 : PARTIAL MATCH WHEN ENUMERATED EXPRESSION IS INCOMPLETE ******/
			String expr2 = _T("A + B , C + D , \t D +");

			// build our chained parser!
			GeneralExpressionParser enumeratedExpr;
			enumeratedExpr << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("+")) << AxisGrammar::CreateIdParser();
			EnumerationParser enumParserCase2(enumeratedExpr);
			it = IteratorFactory::CreateStringIterator(expr2);
			end = IteratorFactory::CreateStringIterator(expr2.end());

			// parse it!
			result = enumParserCase2.Parse(it, end);

			// check if matched
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FullReadPartialMatch, result.GetResult());
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
		}

		TEST_METHOD(TestEnumerationExpressionFailedMatch)
		{
			String expression = _T("A + B , C + D , \t D   -E");

			// build our chained parser!
			GeneralExpressionParser enumeratedExpr;
			enumeratedExpr << AxisGrammar::CreateIdParser() << AxisGrammar::CreateOperatorParser(_T("+")) << AxisGrammar::CreateIdParser();
			EnumerationParser enumeration(enumeratedExpr);
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = enumeration.Parse(it, end);

			// check if matched
			Assert::AreEqual(false, result.IsMatch());
			Assert::AreEqual(ParseResult::FailedMatch, result.GetResult());
			Assert::AreEqual(true, result.GetParseTree().IsEmpty());
		}

		TEST_METHOD(TestComplexGrammar1)
		{
			/*
				We are building the following grammar:
					term ::= id | string | num
					expr ::= term Z
					Z ::= + term Z | epsilon
			*/
			OrExpressionParser term;
			GeneralExpressionParser expr;
			OrExpressionParser Z;
			GeneralExpressionParser auxZ;	// just to express the first expression in production Z

			term << AxisGrammar::CreateIdParser() << AxisGrammar::CreateStringParser() << AxisGrammar::CreateNumberParser();
			expr << term << Z;
			auxZ << AxisGrammar::CreateOperatorParser(_T("+")) << term << Z;
			Z << auxZ << AxisGrammar::CreateEpsilonParser();

			// build expression
			String expression = _T("var1+var2   +   2.5e-8  + \t \"string\" +40");
			String trimmedExpression = _T("var1+var2+2.5e-8+ \"string\" +40");
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = expr.Parse(it, end);

			// check if matched
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(false, result.GetParseTree().IsEmpty());
			Logger::WriteMessage(result.GetParseTree().BuildExpressionString());
			Assert::AreEqual(trimmedExpression, result.GetParseTree().BuildExpressionString()); // that is, an expression
		}

		TEST_METHOD(TestComplexGrammar2)
		{
			/*
				We are building the following grammar:
					array ::= ( enum )
					enum ::= attr | val
					val ::= id | num | str | array
					attr ::= id = val
			*/
			GeneralExpressionParser arrayExpr;
			OrExpressionParser enumerationInnerExpr;
			EnumerationParser enumerationExpr(enumerationInnerExpr);
			OrExpressionParser valueExpr;
			AssignmentParser assignmentExpr;
			arrayExpr << AxisGrammar::CreateOperatorParser(_T("(")) << enumerationExpr << AxisGrammar::CreateOperatorParser(_T(")"));
			enumerationInnerExpr << assignmentExpr << valueExpr;
			valueExpr << AxisGrammar::CreateIdParser() << AxisGrammar::CreateNumberParser() << AxisGrammar::CreateStringParser() << arrayExpr;
			assignmentExpr.SetRhsExpression(valueExpr);

			// build expression
			String expression = _T("(var1 = 2.56 , var2,   2.54, var3 = (1,2,5,7,9), var4 = \"hello\", var5 = test, ( 1, 2, 3, (4 ,5 ,6 ,7 )  ) )");
			String trimmedExpression = _T("(var1=2.56,var2,2.54,var3=(1,2,5,7,9),var4=\"hello\",var5=test,(1,2,3,(4,5,6,7) ) )");
			InputIterator it = IteratorFactory::CreateStringIterator(expression);
			InputIterator end = IteratorFactory::CreateStringIterator(expression.end());

			// parse it!
			ParseResult result = arrayExpr.Parse(it, end);

			// check if matched
			Assert::AreEqual(true, result.IsMatch());
			Assert::AreEqual(false, result.GetParseTree().IsEmpty());
			Assert::AreEqual(end, result.GetLastReadPosition());
			Logger::WriteMessage(result.GetParseTree().BuildExpressionString());
			Assert::AreEqual(trimmedExpression, result.GetParseTree().BuildExpressionString()); // that is, an expression
		}

	};

}

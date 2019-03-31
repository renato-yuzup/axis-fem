#include "ParameterListParser.hpp"
#include "../parsing/SymbolTerminal.hpp"
#include "../parsing/ExpressionNode.hpp"
#include "../parsing/AssignmentExpression.hpp"
#include "foundation/ArgumentException.hpp"
#include "evaluation/ArrayValue.hpp"
#include "../grammar_tokens.hpp"
#include "../parsing/OperatorTerminal.hpp"
#include "../parsing/EnumerationExpression.hpp"
#include "evaluation/ParameterAssignment.hpp"
#include "evaluation/IdValue.hpp"
#include "evaluation/NumberValue.hpp"
#include "evaluation/StringValue.hpp"
#include "../primitives/OrExpressionParser.hpp"
#include "../primitives/EnumerationParser.hpp"
#include "../primitives/AssignmentParser.hpp"
#include "../primitives/GeneralExpressionParser.hpp"
#include "../factories/AxisGrammar.hpp"
#include "evaluation/ParameterListImpl.hpp"
#include "evaluation/NullValue.hpp"

namespace aslf = axis::services::language::factories;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;
namespace asls = axis::services::language::syntax;
namespace aslse = axis::services::language::syntax::evaluation;

asls::ParameterListParser::ParameterListParser( void )
{
	// nothing to do
}

asls::evaluation::ParameterList& asls::ParameterListParser::ParseParameterList( 
  const aslp::EnumerationExpression& parseTree )
{
	aslse::ParameterListImpl *params = NULL;

	try
	{
		params = new evaluation::ParameterListImpl();
		const aslp::ParseTreeNode *node = parseTree.GetFirstChild(); // first child of enumeration node
		while(node != NULL)
		{
			if (node->IsTerminal())	// it might be an id; check it first
			{
        const aslp::SymbolTerminal& symbTerm = *static_cast<const aslp::SymbolTerminal *>(node);
				if (!symbTerm.IsId()) throw axis::foundation::ArgumentException();
				params->AddParameter((static_cast<const aslp::IdTerminal&>(symbTerm)).GetId(), 
                             *new aslse::NullValue());
			}
			else
			{	// if not, then it might be an assignment; check it first
				const aslp::ExpressionNode& exprNode = *static_cast<const aslp::ExpressionNode *>(node);
        if (!exprNode.IsAssignment()) throw axis::foundation::ArgumentException();
				const aslp::AssignmentExpression &assignment = 
          static_cast<const aslp::AssignmentExpression&>(exprNode);
				params->AddParameter(assignment.GetLhs().ToString(), 
                             BuildValueFromParseTree(assignment.GetRhs()));
			}
			node = node->GetNextSibling();
		}
	}
	catch (...)
	{
		if (params) delete params;
		throw;
	}
	return *params;
}

aslse::ParameterValue& asls::ParameterListParser::BuildValueFromParseTree( 
  const aslp::ParseTreeNode& parseTree )
{
	// delegate task to the correct method
	if (parseTree.IsTerminal())
	{
		return BuildAtomicValueFromParseTree(static_cast<const aslp::SymbolTerminal&>(parseTree));
	}
	else
	{
		return BuildExpressionFromParseTree(static_cast<const aslp::ExpressionNode&>(parseTree));	
	}
}

aslse::ParameterValue& asls::ParameterListParser::BuildExpressionFromParseTree( 
  const aslp::ExpressionNode& parseTree )
{
	// delegate task to the correct method
	if (parseTree.IsAssignment())
	{
		return BuildAssignmentFromParseTree(static_cast<const aslp::AssignmentExpression&>(parseTree));
	}
	else if (parseTree.IsRhs())	 // an array
	{
		return BuildArrayFromParseTree(static_cast<const aslp::RhsExpression&>(parseTree));
	}

	// nothing else worked, we don't know how to build it
	throw axis::foundation::ArgumentException();
}

aslse::ParameterValue& asls::ParameterListParser::BuildArrayFromParseTree( 
  const aslp::RhsExpression& parseTree )
{
	aslse::ArrayValue *array = NULL;

	// obtain each part of the expression and ensure that it is correct
	if (!(parseTree.GetChildCount() == 2 || parseTree.GetChildCount() == 3)) throw axis::foundation::ArgumentException();

	if (parseTree.GetChildCount() == 2)
	{	// maybe an empty array, check it
		const aslp::ParseTreeNode *openDelimiter = parseTree.GetFirstChild();
		const aslp::ParseTreeNode *closeDelimiter = openDelimiter->GetNextSibling();

		// check open delimiter
		if (!openDelimiter->IsTerminal()) throw axis::foundation::ArgumentException();
		if (!(static_cast<const aslp::SymbolTerminal *>(openDelimiter))->IsOperator()) 
      throw axis::foundation::ArgumentException();
		if ((static_cast<const aslp::OperatorTerminal *>(openDelimiter))->ToString() 
      != AXIS_GRAMMAR_ARRAY_OPEN_DELIMITER) throw axis::foundation::ArgumentException();

		// check close delimiter
		if (!closeDelimiter->IsTerminal()) throw axis::foundation::ArgumentException();
		if (!(static_cast<const aslp::SymbolTerminal *>(closeDelimiter))->IsOperator()) 
      throw axis::foundation::ArgumentException();
		if ((static_cast<const aslp::OperatorTerminal *>(closeDelimiter))->ToString() != 
      AXIS_GRAMMAR_ARRAY_CLOSE_DELIMITER) throw axis::foundation::ArgumentException();

		// ok, it is an empty array
		return *new evaluation::ArrayValue();
	}

	// if we have three children, then of course these assignments will work
	const aslp::ParseTreeNode *openDelimiter = parseTree.GetFirstChild();
	const aslp::ParseTreeNode *enumeration = openDelimiter->GetNextSibling();
	const aslp::ParseTreeNode *closeDelimiter = enumeration->GetNextSibling();

	// check open delimiter
	if (!openDelimiter->IsTerminal()) throw axis::foundation::ArgumentException();
	if (!(static_cast<const aslp::SymbolTerminal *>(openDelimiter))->IsOperator()) 
    throw axis::foundation::ArgumentException();
	if ((static_cast<const aslp::OperatorTerminal *>(openDelimiter))->ToString() != 
    AXIS_GRAMMAR_ARRAY_OPEN_DELIMITER) throw axis::foundation::ArgumentException();

	// check list items
	if (enumeration->IsTerminal()) throw axis::foundation::ArgumentException();
	if (!(static_cast<const aslp::ExpressionNode *>(enumeration))->IsEnumeration()) 
    throw axis::foundation::ArgumentException();

	// check close delimiter
	if (!closeDelimiter->IsTerminal()) throw axis::foundation::ArgumentException();
	if (!(static_cast<const aslp::SymbolTerminal *>(closeDelimiter))->IsOperator()) 
    throw axis::foundation::ArgumentException();
	if ((static_cast<const aslp::OperatorTerminal *>(closeDelimiter))->ToString() != 
    AXIS_GRAMMAR_ARRAY_CLOSE_DELIMITER) throw axis::foundation::ArgumentException();

	// iterate through list items and add to our node
	try
	{
		array = new aslse::ArrayValue();
		const aslp::ParseTreeNode *node = 
      (static_cast<const aslp::EnumerationExpression *>(enumeration))->GetFirstChild();
		while(node != NULL)
		{
			array->AddValue(BuildValueFromParseTree(*node));
			node = node->GetNextSibling();
		}
	}
	catch (...)
	{
		if (array) delete array;
		throw;
	}
	return *array;
}

aslse::ParameterValue& asls::ParameterListParser::BuildAssignmentFromParseTree( 
  const aslp::AssignmentExpression& parseTree )
{
	return *new aslse::ParameterAssignment(parseTree.GetLhs().ToString(), 
                                         BuildValueFromParseTree(parseTree.GetRhs()));
}

aslse::ParameterValue& asls::ParameterListParser::BuildIdFromParseTree( 
  const aslp::IdTerminal& parseTree )
{
	return *new aslse::IdValue(parseTree.GetId());
}

aslse::ParameterValue& asls::ParameterListParser::BuildNumberFromParseTree( 
  const aslp::NumberTerminal& parseTree )
{
	if (parseTree.IsInteger())
	{
		return *new aslse::NumberValue(parseTree.GetInteger());
	}
	else
	{
		return *new aslse::NumberValue(parseTree.GetDouble());
	}
}

aslse::ParameterValue& asls::ParameterListParser::BuildStringFromParseTree( 
  const aslp::StringTerminal& parseTree )
{
	return *new aslse::StringValue(parseTree.ToString());
}

aslse::ParameterValue& asls::ParameterListParser::BuildAtomicValueFromParseTree( 
  const aslp::SymbolTerminal& parseTree )
{
	// delegate task to the correct method
	if (parseTree.IsId())
	{
		return BuildIdFromParseTree(static_cast<const aslp::IdTerminal&>(parseTree));
	}
	else if (parseTree.IsNumber())
	{
		return BuildNumberFromParseTree(static_cast<const aslp::NumberTerminal&>(parseTree));
	}
	else if (parseTree.IsString())
	{
		return BuildStringFromParseTree(static_cast<const aslp::StringTerminal&>(parseTree));
	}

	// if nothing else worked, we don't know how to build it
	throw axis::foundation::ArgumentException();
}

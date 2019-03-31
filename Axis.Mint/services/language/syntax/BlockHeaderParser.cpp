#include "BlockHeaderParser.hpp"
#include "foundation/ArgumentException.hpp"
#include "../factories/AxisGrammar.hpp"
#include "../parsing/OperatorTerminal.hpp"
#include "evaluation/StringValue.hpp"
#include "evaluation/NumberValue.hpp"
#include "evaluation/IdValue.hpp"
#include "evaluation/ArrayValue.hpp"
#include "evaluation/ParameterListImpl.hpp"
#include "evaluation/NullValue.hpp"
#include "evaluation/ParameterAssignment.hpp"
#include "../grammar_tokens.hpp"

namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;
namespace asls = axis::services::language::syntax;
namespace aslse = axis::services::language::syntax::evaluation;

asls::BlockHeaderParser::BlockHeaderParser( void )
{
	/*
		These guys we are going to use to parse the following grammar:

		paramList ::= enumParam [,enumParam]* | epsilon
		enumParam ::= assign | id
		assign ::= id = value
		value ::= id | num | str | array
		array ::= ( arrayContents )
		arrayContents ::= arrayEnum | epsilon
		arrayEnum ::= arrayVal [,arrayVal]* | epsilon
		arrayVal ::= assign | value
	*/
	_enumParam     = new aslpp::OrExpressionParser();
	_paramList     = new aslpp::EnumerationParser(*_enumParam, true);
	_assign        = new aslpp::AssignmentParser();
  _value         = new aslpp::OrExpressionParser();
  _array         = new aslpp::GeneralExpressionParser();
	_arrayVal      = new aslpp::OrExpressionParser();
	_arrayContents = new aslpp::OrExpressionParser();
	_arrayEnum     = new aslpp::EnumerationParser(*_arrayVal);

	*_enumParam << *_assign << aslf::AxisGrammar::CreateIdParser();
	_assign->SetRhsExpression(*_value);
	*_value << aslf::AxisGrammar::CreateIdParser() << aslf::AxisGrammar::CreateNumberParser() << 
		     aslf::AxisGrammar::CreateStringParser() << *_array;
	*_array << aslf::AxisGrammar::CreateOperatorParser(AXIS_GRAMMAR_ARRAY_OPEN_DELIMITER)
		    << *_arrayContents 
		    << aslf::AxisGrammar::CreateOperatorParser(AXIS_GRAMMAR_ARRAY_CLOSE_DELIMITER);
	*_arrayContents << *_arrayEnum << aslf::AxisGrammar::CreateEpsilonParser();
	*_arrayVal << *_assign << *_value;
	
	/*
		And these we define the grammar for the BEGIN...WITH block
	*/
	_blockDeclaration	   = new aslpp::GeneralExpressionParser();
	_blockHeaderWithParams = new aslpp::GeneralExpressionParser();
	*_blockDeclaration <<aslf:: AxisGrammar::CreateBlankParser()  // to remote trailing spaces
		<< aslf::AxisGrammar::CreateReservedWordBlockHeadParser()
		<< aslf::AxisGrammar::CreateBlankParser(true) // ensure there is a space between the reserved word and the name
		<< aslf::AxisGrammar::CreateIdParser();
	*_blockHeaderWithParams << *_blockDeclaration 
    << aslf::AxisGrammar::CreateBlankParser(true)
    << aslf::AxisGrammar::CreateReservedWordParameterListStarterParser()
    << aslf::AxisGrammar::CreateBlankParser(true);
	

	// Initialize parameter list
	_paramListResult = new aslse::ParameterListImpl();
}

asls::BlockHeaderParser::~BlockHeaderParser( void )
{	// release resources
	delete _enumParam;
	delete _paramList;
	delete _assign;
	delete _value;
	delete _array;
	delete _arrayVal;
	delete _arrayContents;
	delete _arrayEnum;
	delete _blockDeclaration;
	delete _blockHeaderWithParams;
	delete _paramListResult;
}

aslp::ParseResult asls::BlockHeaderParser::ParseOnlyParameterList( const asli::InputIterator& begin, 
                                                                   const asli::InputIterator& end )
{
	aslp::ParseResult result = (*_paramList)(begin, end);

	// update to the last read position
	if (result.IsMatch())
	{	// ok, return a complete parameter tree
		CreateParameterListFromParseTree((aslp::EnumerationExpression&)result.GetParseTree());
	}
	else
	{	// not good, return an empty list
		StoreParameterList(*new aslse::ParameterListImpl());		
	}
	return result;
}

void asls::BlockHeaderParser::CreateParameterListFromParseTree( 
  const aslp::EnumerationExpression& parseTree )
{
	aslse::ParameterListImpl *params = NULL;

	try
	{
		params = new aslse::ParameterListImpl();
		const aslp::ParseTreeNode *node = parseTree.GetFirstChild();	// first child of enumeration node
		while(node != NULL)
		{
			if (node->IsTerminal())	// it might be an id; check it first
			{
				if (!(static_cast<const aslp::SymbolTerminal *>(node))->IsId()) 
          throw axis::foundation::ArgumentException();
				params->AddParameter((static_cast<const aslp::IdTerminal *>(node))->GetId(), 
                             *new aslse::NullValue());
			}
			else
			{	// if not, then it might be an assignment; check it first
				if (!(static_cast<const aslp::ExpressionNode *>(node))->IsAssignment()) 
          throw axis::foundation::ArgumentException();
				const aslp::AssignmentExpression *assignment = 
          static_cast<const aslp::AssignmentExpression *>(node);
				params->AddParameter(assignment->GetLhs().ToString(), 
                             BuildValueFromParseTree(assignment->GetRhs()));
			}
			node = node->GetNextSibling();
		}
	}
	catch (...)
	{
		if (params) delete params;
		throw;
	}

	StoreParameterList(*params);
}

aslse::ParameterValue& asls::BlockHeaderParser::BuildValueFromParseTree( 
  const aslp::ParseTreeNode& parseTree ) const
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

aslse::ParameterValue& asls::BlockHeaderParser::BuildExpressionFromParseTree( 
  const aslp::ExpressionNode& parseTree ) const
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

aslse::ParameterValue& asls::BlockHeaderParser::BuildArrayFromParseTree( 
  const aslp::RhsExpression& parseTree ) const
{
	evaluation::ArrayValue *array = NULL;
  int childCount = parseTree.GetChildCount();

	// obtain each part of the expression and ensure that it is correct
	if (!(childCount == 2 || childCount == 3)) 
    throw axis::foundation::ArgumentException();

	const aslp::ParseTreeNode *openDelimiter = parseTree.GetFirstChild();
	const aslp::ParseTreeNode *closeDelimiter = openDelimiter->GetNextSibling();
  if (childCount == 3)
  {
    closeDelimiter = closeDelimiter->GetNextSibling();
  }

	// check open delimiter
	if (!openDelimiter->IsTerminal()) throw axis::foundation::ArgumentException();
  const aslp::SymbolTerminal &openSymbTerm = 
    *static_cast<const aslp::SymbolTerminal *>(openDelimiter);
	if (!openSymbTerm.IsOperator()) 
    throw axis::foundation::ArgumentException();
	if (openSymbTerm.ToString() != AXIS_GRAMMAR_ARRAY_OPEN_DELIMITER) 
    throw axis::foundation::ArgumentException();

	// check close delimiter
	if (!closeDelimiter->IsTerminal()) throw axis::foundation::ArgumentException();
  const aslp::SymbolTerminal& closeSymbTerm = 
    *static_cast<const aslp::SymbolTerminal *>(closeDelimiter);
	if (!closeSymbTerm.IsOperator()) throw axis::foundation::ArgumentException();
	if (closeSymbTerm.ToString() != AXIS_GRAMMAR_ARRAY_CLOSE_DELIMITER) 
    throw axis::foundation::ArgumentException();

  if (parseTree.GetChildCount() == 2)
	{	
		// ok, it is an empty array
		return *new aslse::ArrayValue();
	}

	// if we have three children, then of course these assignments will work
	const aslp::ParseTreeNode *enumeration = openDelimiter->GetNextSibling();

	// check list items
	if (enumeration->IsTerminal()) throw axis::foundation::ArgumentException();
	if (!(static_cast<const aslp::ExpressionNode *>(enumeration))->IsEnumeration()) 
    throw axis::foundation::ArgumentException();

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

aslse::ParameterValue& asls::BlockHeaderParser::BuildAssignmentFromParseTree( 
  const aslp::AssignmentExpression& parseTree ) const
{
	return *new evaluation::ParameterAssignment(parseTree.GetLhs().ToString(), BuildValueFromParseTree(parseTree.GetRhs()));
}

aslse::ParameterValue& asls::BlockHeaderParser::BuildIdFromParseTree( 
  const aslp::IdTerminal& parseTree ) const
{
	return *new evaluation::IdValue(parseTree.GetId());
}

aslse::ParameterValue& asls::BlockHeaderParser::BuildNumberFromParseTree( 
  const aslp::NumberTerminal& parseTree ) const
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

aslse::ParameterValue& asls::BlockHeaderParser::BuildStringFromParseTree( 
  const aslp::StringTerminal& parseTree ) const
{
	return *new evaluation::StringValue(parseTree.ToString());
}

aslse::ParameterValue& asls::BlockHeaderParser::BuildAtomicValueFromParseTree( 
  const aslp::SymbolTerminal& parseTree ) const
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

const aslse::ParameterList& asls::BlockHeaderParser::GetParameterList( void ) const
{
	return *_paramListResult;
}

axis::String asls::BlockHeaderParser::GetBlockName( void ) const
{
	return _blockName;
}

void asls::BlockHeaderParser::StoreParameterList( aslse::ParameterList& result )
{
	delete _paramListResult;
	_paramListResult = &result;
}

void asls::BlockHeaderParser::ClearParameterListResult( void )
{
	StoreParameterList(*new evaluation::ParameterListImpl());
}

bool asls::BlockHeaderParser::HasParameters( void ) const
{
	return !_paramListResult->IsEmpty();
}

aslp::ParseResult asls::BlockHeaderParser::Parse( const asli::InputIterator& begin, 
                                                  const asli::InputIterator& end )
{
	String blockName;


	// check if we have parameters
  aslp::ParseResult resultWithParams = (*_blockHeaderWithParams)(begin, end, false);
  aslp::ParseResult resultNoParams = (*_blockDeclaration)(begin, end, false);


	if ((resultNoParams.IsMatch() && !(resultWithParams.IsMatch() || 
    resultWithParams.GetResult() == aslp::ParseResult::FullReadPartialMatch)) ||
		(resultNoParams.IsMatch() && resultNoParams.GetLastReadPosition() == end))
	{	// a block header without parameters
		aslp::ExpressionNode& declaration = (aslp::ExpressionNode&)resultNoParams.GetParseTree();
		aslp::ParseTreeNode& idBlock = *declaration.GetFirstChild()->GetNextSibling();
		blockName = idBlock.ToString();

		ClearParameterListResult();
		_blockName = blockName;

		return resultNoParams;
	}
	else if (resultWithParams.IsMatch())
	{	// a block header with parameters
		if (resultWithParams.GetLastReadPosition() == end)
		{	// hey, we can't read the parameters yet, it's just a partial match
			resultWithParams.SetResult(aslp::ParseResult::FullReadPartialMatch);
			return resultWithParams;
		}

		// get block name before we overwrite results
		aslp::ExpressionNode& rootNode = ((aslp::ExpressionNode&)resultWithParams.GetParseTree());
		aslp::ExpressionNode& declaration = (aslp::ExpressionNode&)*rootNode.GetFirstChild();
		aslp::ParseTreeNode& idBlock = *declaration.GetFirstChild()->GetNextSibling();
		blockName = idBlock.ToString();

		// parse it!
		resultWithParams = ParseOnlyParameterList(resultWithParams.GetLastReadPosition(), end);
		_blockName = resultWithParams.IsMatch()? blockName : _T("");		
		return resultWithParams;
	}
	else if (resultWithParams.GetResult() == aslp::ParseResult::FullReadPartialMatch)
	{	// either a block header with or without parameters, but still incomplete
		ClearParameterListResult();
		_blockName = _T("");
		return resultWithParams;
	}
	// we have a no-match situation
	ClearParameterListResult();
	_blockName = _T("");
	return resultWithParams;	
}

aslp::ParseResult asls::BlockHeaderParser::operator()( const asli::InputIterator& begin, 
                                                       const asli::InputIterator& end )
{
	return Parse(begin, end);
}

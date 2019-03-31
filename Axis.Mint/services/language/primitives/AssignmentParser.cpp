#include "AssignmentParser.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "../factories/AxisGrammar.hpp"
#include "AtomicExpressionParser.hpp"
#include "../parsing/AssignmentExpression.hpp"
#include "../parsing/IdTerminal.hpp"
#include "../parsing/RhsExpression.hpp"

namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;

aslpp::AssignmentParser::AssignmentParser( void )
{
	_parser.Add(aslf::AxisGrammar::CreateIdParser())
	       .Add(aslf::AxisGrammar::CreateBlankParser())
		     .Add(aslf::AxisGrammar::CreateOperatorParser(_T("=")))
		     .Add(aslf::AxisGrammar::CreateBlankParser());
	_rhs = NULL;
	_isNullExpression = true;
	try
	{
		_rhs = new AtomicExpressionParser(aslf::AxisGrammar::CreateBlankParser());
		_parser.Add(*_rhs);
	}
	catch (std::bad_alloc)
	{	
		if(_rhs) delete _rhs;
		// throw an exception
		throw axis::foundation::OutOfMemoryException();
	}
}

aslpp::AssignmentParser::~AssignmentParser( void )
{
	if (_isNullExpression)
	{
		delete _rhs;
	}
}

aslp::ParseResult aslpp::AssignmentParser::DoParse( const asli::InputIterator& begin, 
                                                    const asli::InputIterator& end, 
                                                    bool trimSpaces /*= true*/ ) const
{
	aslp::ParseResult result = _parser.Parse(begin, end, trimSpaces);
	if (result.IsMatch())
	{	// ok, append elements to our own node and return it
		aslp::RhsExpression& rootNode = (aslp::RhsExpression&)result.GetParseTree();
		aslp::IdTerminal& idResultNode = (aslp::IdTerminal&)*rootNode.GetFirstChild();
		aslp::ParseTreeNode& rhsNode = *idResultNode.GetNextSibling()->GetNextSibling();	// the node after the assign operator
		aslp::AssignmentExpression& node = *new aslp::AssignmentExpression(idResultNode.Clone(), rhsNode.Clone());
		return aslp::ParseResult(result.GetResult(), node, result.GetLastReadPosition());
	}
	// no full match -- return the same result obtained
	return result;
}

aslpp::CompositeParser& aslpp::AssignmentParser::Add( ExpressionParser& expression )
{
	throw axis::foundation::NotSupportedException();
}

aslpp::CompositeParser& aslpp::AssignmentParser::Add( const Parser& parser )
{
	throw axis::foundation::NotSupportedException();
}

void aslpp::AssignmentParser::Remove( const ExpressionParser& expression )
{
	throw axis::foundation::NotSupportedException();
}

void aslpp::AssignmentParser::SetRhsExpression( const ExpressionParser& expression )
{
	// first, add new RHS (can throw an exception)
	CompositeParser::Add(expression);
	// now, remove old expression (exception free)
	if (_isNullExpression)
	{
		delete _rhs;
		_isNullExpression = false;
	}
	_parser.Remove(*_rhs);
	// assign new RHS
	_rhs = &expression;
	_parser.Add(expression);
}

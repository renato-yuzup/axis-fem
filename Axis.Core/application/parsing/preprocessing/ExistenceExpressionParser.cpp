#include "ExistenceExpressionParser.hpp"
#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/InvalidIdentifierException.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/OperatorTerminal.hpp"
#include "services/language/parsing/ExpressionNode.hpp"

namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace afd  = axis::foundation::definitions;
namespace aapp = axis::application::parsing::preprocessing;

/**********************************************************************************************//**
 * @summary We will use these tags to differ the many operators.
 **************************************************************************************************/
namespace {
const int OpenDelimiterTag	= 0xF0;
const int CloseDelimiterTag	= 0xFF;
const int BinaryOperatorTag = afd::kBinaryOperator;
const int UnaryOperatorTag	= afd::kUnaryOperator;
}

aapp::ExistenceExpressionParser::ExistenceExpressionParser(aapp::SymbolTable& st) : _symbolTable(st)
{
	_lastResult = false;

	// initialize grammar rules
	_binary_op			      << aslf::AxisGrammar::CreateOperatorParser(
                            afd::AxisInputLanguage::Operators.OperatorAnd.GetSyntax(), 
                            BinaryOperatorTag, 
                            afd::AxisInputLanguage::Operators.OperatorAnd.GetPrecedence(), 
                            afd::AxisInputLanguage::Operators.OperatorAnd.GetAssociativity())
						            << aslf::AxisGrammar::CreateOperatorParser(
                            afd::AxisInputLanguage::Operators.OperatorOr.GetSyntax(), 
                            BinaryOperatorTag, 
                            afd::AxisInputLanguage::Operators.OperatorOr.GetPrecedence(), 
                            afd::AxisInputLanguage::Operators.OperatorOr.GetAssociativity());
	_unary_op			        << aslf::AxisGrammar::CreateOperatorParser(
                            afd::AxisInputLanguage::Operators.OperatorNot.GetSyntax(), 
                            UnaryOperatorTag, 
                            afd::AxisInputLanguage::Operators.OperatorNot.GetPrecedence(), 
                            afd::AxisInputLanguage::Operators.OperatorNot.GetAssociativity());
	_expression			      << _expression_alt1 << _term;
	_expression_alt1	    << _term << _expression2;
	_expression2		      << _expression2_alt1 << _expression2_alt2;
	_expression2_alt1	    << _binary_op << _term << _expression2;
	_expression2_alt2	    << _binary_op << _term;
	_term				          << _term_alt1 << _operand;
	_term_alt1			      << _unary_op << _operand;
	_operand			        << aslf::AxisGrammar::CreateIdParser() << _group;
	_group				        << aslf::AxisGrammar::CreateOperatorParser(
                            afd::AxisInputLanguage::Operators.OpenGroup.GetSyntax(), 
                            OpenDelimiterTag) << _expression 
                        << aslf::AxisGrammar::CreateOperatorParser(
                            afd::AxisInputLanguage::Operators.CloseGroup.GetSyntax(), 
                            CloseDelimiterTag);
	_invalidIdExpression	<< _binary_op << _unary_op;
}

aapp::ExistenceExpressionParser::~ExistenceExpressionParser(void)
{
	// clear token list
	_expressionTokens.clear();
}

void aapp::ExistenceExpressionParser::EvaluateParseTree( const aslp::ParseTreeNode & parseNode ) 
{
	if (parseNode.IsTerminal())
	{
		aslp::SymbolTerminal& terminal = (aslp::SymbolTerminal&)parseNode;
		if(terminal.IsId())
		{	// it is supposed to be an ID, check if it doesn't conflict with
			// any reserved keywords
			if (!IsValidId(terminal.ToString()))
			{	// conflict detected
				throw axis::foundation::InvalidIdentifierException();
			}
			// add to the output list
			_expressionTokens.push_back(Symbol(afd::kId, terminal.ToString(), 0, afd::kLeftAssociativity));
		}
		else if(terminal.IsOperator())
		{	// it is an operator or a delimiter
			aslp::OperatorTerminal& op = (aslp::OperatorTerminal&)terminal;
			if (op.GetValue() == BinaryOperatorTag || op.GetValue() == UnaryOperatorTag)
			{	// in fact it is an operator
				if (_operatorStack.size() > 0)
				{	// there are operators in the stack; we might need to pop them out
					aapp::Symbol topOperator = _operatorStack.front();
					while(topOperator.Type != afd::kDelimiter &&            // it is not a delimiter
						    (((afd::OperatorAssociativity)op.GetAssociativity() == afd::kLeftAssociativity && 
                op.GetPrecedence() <= topOperator.Precedence) ||	// on stack: left-associative and 
                                                                  // takes precedence (or same 
                                                                  // precedence) over our operator
						    ((afd::OperatorAssociativity)op.GetAssociativity() != afd::kLeftAssociativity && 
                op.GetPrecedence() < topOperator.Precedence)))		// on stack: right-associative and 
                                                                  // takes precedence over our operator
					{
						// send to the output queue
						_expressionTokens.push_back(topOperator);
						// pop out top element
						_operatorStack.pop_front();
						if (_operatorStack.size() == 0) break;
						topOperator = _operatorStack.front();
					}
				}
				// put in the operator stack
				_operatorStack.push_front(Symbol((afd::TokenType)op.GetValue(), op.ToString(), 
          op.GetPrecedence(), (afd::OperatorAssociativity)op.GetAssociativity()));
			}
			else if(op.GetValue() == OpenDelimiterTag)
			{	// it is a left delimiter, put in the stack
				_operatorStack.push_front(Symbol(afd::kDelimiter, op.ToString(), op.GetPrecedence(), 
          (afd::OperatorAssociativity)op.GetAssociativity()));
			}
			else if(op.GetValue() == CloseDelimiterTag)
			{	// it is a right delimiter, pop out tokens
				if (_operatorStack.size() == 0)
				{	// something wrong here...
					throw axis::foundation::InvalidSyntaxException();
				}
				Symbol topOperator = _operatorStack.front();
				while(!(topOperator.Name.compare(afd::AxisInputLanguage::Operators.OpenGroup.GetSyntax()) == 0 
              && topOperator.Type == afd::kDelimiter))
				{
					// send to the output queue
					_expressionTokens.push_back(topOperator);
					// pop out top element
					_operatorStack.pop_front();
					if (_operatorStack.size() == 0)
					{	// the stack was emptied without finding a left delimiter...
						throw axis::foundation::InvalidSyntaxException();
					}
					topOperator = _operatorStack.front();
				}
				// remove open delimiter
				_operatorStack.pop_front();
			}
		}
	}
	else
	{	// it's an expression; go to first child and continue preprocessing from there
		aslp::ExpressionNode& expression = (aslp::ExpressionNode&)parseNode;
		EvaluateParseTree(*expression.GetFirstChild());
	}

	// check if there's another sibling to process
	if (parseNode.GetNextSibling() != NULL)
	{
		EvaluateParseTree(*parseNode.GetNextSibling());
	}
}

bool aapp::ExistenceExpressionParser::IsSyntacticallyValid(const axis::String& e)
{
	using axis::String;

	// clear symbol table before start evaluating
	_expressionTokens.clear();

	// try to Evaluate expression
	asli::InputIterator it = aslf::IteratorFactory::CreateStringIterator(e.begin());
	asli::InputIterator end = aslf::IteratorFactory::CreateStringIterator(e.end());
	aslp::ParseResult result = _expression(it, end);

	if (!result.IsMatch())
	{	// invalid syntax
		return false;
	}
	// evaluate parse result
	aslp::ParseTreeNode & parseRoot = result.GetParseTree();
	EvaluateParseTree(parseRoot);
	// push remaining operators from the stack
	while(_operatorStack.size() > 0)
	{
		Symbol s = _operatorStack.front();
		if(s.Name.compare(afd::AxisInputLanguage::Operators.OpenGroup.GetSyntax()) == 0 && 
       s.Type == afd::AxisInputLanguage::Operators.OpenGroup.GetType())
		{	// huh? another open delimiter...
			throw axis::foundation::InvalidSyntaxException();
		}
		_expressionTokens.push_back(s);
		_operatorStack.pop_front();
	}
	return result.IsMatch(); // that is, a full match
}

void aapp::ExistenceExpressionParser::ProcessToken( afd::TokenType type, axis::String const& name )
{
	ProcessToken(type, name, 0, afd::kLeftAssociativity);
}

void aapp::ExistenceExpressionParser::ProcessToken( afd::TokenType type, axis::String const& name, 
  int precedence, afd::OperatorAssociativity associativity )
{
	/*
		We will use the shunting-yard algorithm to put the expression in
		the RPM format so that it can be easily evaluated.
	*/
	if (type == afd::kId)
	{	// it is supposed to be an ID, check if it doesn't conflict with
		// any reserved keywords
		if (!IsValidId(name))
		{	// conflict detected
			throw axis::foundation::InvalidIdentifierException();
		}

		// add to the output list
		_expressionTokens.push_back(Symbol(type, name, precedence, associativity));
	}
	else if(type == afd::kBinaryOperator || type == afd::kUnaryOperator)
	{	// it is an operator
		if (_operatorStack.size() > 0)
		{	// there are operators in the stack; we might need to pop them out
			Symbol topOperator = _operatorStack.front();
			while(topOperator.Type == afd::kDelimiter &&	// it is not a delimiter
				((associativity == afd::kLeftAssociativity && 
        precedence <= topOperator.Precedence) ||	  // on stack: left-associative and takes 
                                                    // precedence (or same precedence) over our operator
				(associativity != afd::kLeftAssociativity && 
        precedence < topOperator.Precedence)))		  // on stack: right-associative and takes 
                                                    // precedence over our operator
			{
				// send to the output queue
				_expressionTokens.push_back(topOperator);
				// pop out top element
				_operatorStack.pop_front();
				if (_operatorStack.size() == 0) break;
				topOperator = _operatorStack.front();
			}
		}
		// put in the operator stack
		_operatorStack.push_front(Symbol(type, name, precedence, associativity));
	}
	else if(type == afd::kDelimiter && name == afd::AxisInputLanguage::Operators.OpenGroup.GetSyntax())
	{	// it is a left delimiter, put in the stack
		_operatorStack.push_front(Symbol(type, name, precedence, associativity));
	}
	else if(type == afd::kDelimiter && name == afd::AxisInputLanguage::Operators.CloseGroup.GetSyntax())
	{	// it is a right delimiter, pop out tokens
		if (_operatorStack.size() == 0)
		{	// something wrong here...
			throw axis::foundation::InvalidSyntaxException();
		}
		Symbol topOperator = _operatorStack.front();
		while(!(topOperator.Name == afd::AxisInputLanguage::Operators.OpenGroup.GetSyntax() && 
            topOperator.Type == afd::kDelimiter))
		{
			// send to the output queue
			_expressionTokens.push_back(topOperator);
			// pop out top element
			_operatorStack.pop_front();
			if (_operatorStack.size() == 0) break;
			topOperator = _operatorStack.front();
		}
		if (_operatorStack.size() == 0)
		{	// the stack was emptied without finding a left delimiter...
			throw axis::foundation::InvalidSyntaxException();
		}
		_operatorStack.pop_front();	// remove left delimiter
	}
}

bool aapp::ExistenceExpressionParser::IsValidId( const axis::String& id ) const
{
	asli::InputIterator it   = aslf::IteratorFactory::CreateStringIterator(id.begin());
	asli::InputIterator end  = aslf::IteratorFactory::CreateStringIterator(id.end());
	aslp::ParseResult result = _invalidIdExpression(it, end);

	return !result.IsMatch();
}

bool aapp::ExistenceExpressionParser::GetLastResult( void ) const
{
	return _lastResult;
}

bool aapp::ExistenceExpressionParser::Evaluate( const axis::String& e )
{
	return Evaluate(e, true);
}

bool aapp::ExistenceExpressionParser::Evaluate( const axis::String& e, bool expectedSymbolDefinedState )
{
	std::list<bool> workStack;
	if (!IsSyntacticallyValid(e))
	{	// syntax error
		throw axis::foundation::InvalidSyntaxException();
	}
	// Evaluate according to the symbol table
	for(token_list::iterator it = _expressionTokens.begin(); it != _expressionTokens.end(); ++it)
	{
		Symbol& s = *it;
		if (s.Type == afd::kId)
		{	// it is an identifier, check if it exists
			workStack.push_back(_symbolTable.IsDefined(s.Name) == expectedSymbolDefinedState);
		}
		else if(s.Type == afd::kBinaryOperator || s.Type == afd::kUnaryOperator)
		{	// it is an operator, pop operands from stack and Evaluate
			if (s.Name == afd::AxisInputLanguage::Operators.OperatorNot.GetSyntax())
			{
				bool operand = workStack.front();
				workStack.pop_front();
				workStack.push_front(!operand);
			}
			else if (s.Name == afd::AxisInputLanguage::Operators.OperatorAnd.GetSyntax())
			{
				bool op2 = workStack.front();
				workStack.pop_front();
				bool op1 = workStack.front();
				workStack.pop_front();
				workStack.push_front(op1 && op2);
			}
			else if (s.Name == afd::AxisInputLanguage::Operators.OperatorOr.GetSyntax())
			{
				bool op2 = workStack.front();
				workStack.pop_front();
				bool op1 = workStack.front();
				workStack.pop_front();
				workStack.push_front(op1 || op2);
			}
		}
	}
	// retrieve result from the stack
	_lastResult = workStack.front();
	return true;
}
#pragma once
#include "AxisString.hpp"
#include "SymbolTable.hpp"
#include <boost/spirit/include/qi.hpp>
#include <list>
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"

namespace axis { namespace application { namespace parsing { namespace preprocessing {

/// <summary>
/// Defines an existence expression parser with the default
/// behavior.
/// </summary>
class ExistenceExpressionParser
{
public:
	/// <summary>
	/// Creates a new instance of this class.
	/// </summary>
	/// <param name="st">Associated symbol table to be used when evaluating an expression.</param>
	ExistenceExpressionParser(axis::application::parsing::preprocessing::SymbolTable& st);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	~ExistenceExpressionParser(void);

	/// <summary>
	/// Returns if the supplied expression is syntactically valid.
	/// </summary>
	/// <param name="e">The expression to be validated.</param>
	bool IsSyntacticallyValid(const axis::String& e);

	/// <summary>
	/// Evaluates the supplied expression.
	/// </summary>
	/// <param name="e">The expression to be evaluated.</param>
	/// <returns>
	/// Returns True if preprocessing was successful, False otherwise.
	/// </returns>
	bool Evaluate(const axis::String& e);

	/// <summary>
	/// Evaluates the supplied expression.
	/// </summary>
	/// <param name="e">The expression to be evaluated.</param>
	/// <param name="expectedSymbolDefinedState">State (defined or not) expected when evaluating each identifier in expression.</param>
	/// <returns>
	/// Returns True if preprocessing was successful, False otherwise.
	/// </returns>
	bool Evaluate(const axis::String& e, bool expectedSymbolDefinedState);

	/// <summary>
	/// Returns the result of the last evaluated expression.
	/// </summary>
	/// <remarks>
	/// If no expression has been evaluated by this object yet, False is returned.
	/// </remarks>
	bool GetLastResult( void ) const;
private:
  // our temporary work list of tokens
  typedef std::list<axis::application::parsing::preprocessing::Symbol> token_list;

  /// <summary>
  /// Process a token obtained when parsing an expression.
  /// </summary>
  /// <param name="type">Type identifier of the token.</param>
  /// <param name="name">Token name.</param>
  void ProcessToken(axis::foundation::definitions::TokenType type, axis::String const& name);

  /// <summary>
  /// Process a token obtained when parsing an expression.
  /// </summary>
  /// <param name="type">Type identifier of the token.</param>
  /// <param name="name">Token name.</param>
  /// <param name="precedence">Precedence number of the token.</param>
  /// <param name="associativity">Associativity number of the token.</param>
  void ProcessToken(axis::foundation::definitions::TokenType type, axis::String const& name, 
    int precedence, axis::foundation::definitions::OperatorAssociativity associativity);
  void EvaluateParseTree(const axis::services::language::parsing::ParseTreeNode & parseNode);
  bool IsValidId(const axis::String& id) const;

  // our grammar rules
  axis::services::language::primitives::OrExpressionParser _binary_op;
  axis::services::language::primitives::OrExpressionParser _unary_op;
  axis::services::language::primitives::OrExpressionParser _expression;
  axis::services::language::primitives::GeneralExpressionParser _expression_alt1;
  axis::services::language::primitives::OrExpressionParser _expression2;
  axis::services::language::primitives::GeneralExpressionParser _expression2_alt1;
  axis::services::language::primitives::GeneralExpressionParser _expression2_alt2;
  axis::services::language::primitives::OrExpressionParser _term;
  axis::services::language::primitives::GeneralExpressionParser _term_alt1;
  axis::services::language::primitives::OrExpressionParser _operand;
  axis::services::language::primitives::GeneralExpressionParser _group;
  axis::services::language::primitives::OrExpressionParser _invalidIdExpression;

  // associated symbol table
  axis::application::parsing::preprocessing::SymbolTable& _symbolTable;

  // used when evaluating an expression
  token_list _expressionTokens;
  token_list _operatorStack;
  bool _lastResult;
};

} } } } // namespace axis::application::parsing::preprocessing

#pragma once
#include "foundation/Axis.Mint.hpp"
#include "evaluation/ParameterValue.hpp"
#include "../parsing/ParseTreeNode.hpp"
#include "../parsing/ExpressionNode.hpp"
#include "../parsing/RhsExpression.hpp"
#include "../parsing/AssignmentExpression.hpp"
#include "../parsing/SymbolTerminal.hpp"
#include "../parsing/IdTerminal.hpp"
#include "../parsing/NumberTerminal.hpp"
#include "../parsing/StringTerminal.hpp"
#include "evaluation/ParameterList.hpp"
#include "../parsing/EnumerationExpression.hpp"

namespace axis { namespace services { namespace language { namespace syntax {

class AXISMINT_API ParameterListParser
{
public:
	static axis::services::language::syntax::evaluation::ParameterList& ParseParameterList(
    const axis::services::language::parsing::EnumerationExpression& parseTree);
private:
  static axis::services::language::syntax::evaluation::ParameterValue& BuildValueFromParseTree(
    const axis::services::language::parsing::ParseTreeNode& parseTree);
  static axis::services::language::syntax::evaluation::ParameterValue& BuildExpressionFromParseTree(
    const axis::services::language::parsing::ExpressionNode& parseTree);
  static axis::services::language::syntax::evaluation::ParameterValue& BuildArrayFromParseTree(
    const axis::services::language::parsing::RhsExpression& parseTree);
  static axis::services::language::syntax::evaluation::ParameterValue& BuildAssignmentFromParseTree(
    const axis::services::language::parsing::AssignmentExpression& parseTree);
  static axis::services::language::syntax::evaluation::ParameterValue& BuildAtomicValueFromParseTree(
    const axis::services::language::parsing::SymbolTerminal& parseTree);					
  static axis::services::language::syntax::evaluation::ParameterValue& BuildIdFromParseTree(
    const axis::services::language::parsing::IdTerminal& parseTree);
  static axis::services::language::syntax::evaluation::ParameterValue& BuildNumberFromParseTree(
    const axis::services::language::parsing::NumberTerminal& parseTree);
  static axis::services::language::syntax::evaluation::ParameterValue& BuildStringFromParseTree(
    const axis::services::language::parsing::StringTerminal& parseTree);

	// with this, there is no way to instantiate this class
	ParameterListParser(void);
};

} } } } // namespace axis::services::language::syntax

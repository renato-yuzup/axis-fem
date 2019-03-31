#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../iterators/InputIterator.hpp"
#include "evaluation/ParameterList.hpp"
#include "../parsing/EnumerationExpression.hpp"
#include "../parsing/RhsExpression.hpp"
#include "../parsing/AssignmentExpression.hpp"
#include "../parsing/IdTerminal.hpp"
#include "../parsing/NumberTerminal.hpp"
#include "../parsing/StringTerminal.hpp"
#include "../parsing/ParseResult.hpp"
#include "../primitives/OrExpressionParser.hpp"
#include "../primitives/EnumerationParser.hpp"
#include "../primitives/AssignmentParser.hpp"
#include "../primitives/GeneralExpressionParser.hpp"

namespace axis { namespace services { namespace language { namespace syntax {

class AXISMINT_API BlockHeaderParser
{
public:
	BlockHeaderParser(void);
	~BlockHeaderParser(void);

	const evaluation::ParameterList& GetParameterList(void) const;
	bool HasParameters(void) const;
	axis::String GetBlockName(void) const;

	axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	axis::services::language::parsing::ParseResult operator()(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	axis::services::language::parsing::ParseResult ParseOnlyParameterList(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	void CreateParameterListFromParseTree(
    const axis::services::language::parsing::EnumerationExpression& parseTree);
private:
  evaluation::ParameterValue& BuildValueFromParseTree(
    const axis::services::language::parsing::ParseTreeNode& parseTree) const;
  evaluation::ParameterValue& BuildExpressionFromParseTree(
    const axis::services::language::parsing::ExpressionNode& parseTree) const;
  evaluation::ParameterValue& BuildArrayFromParseTree(
    const axis::services::language::parsing::RhsExpression& parseTree) const;
  evaluation::ParameterValue& BuildAssignmentFromParseTree(
    const axis::services::language::parsing::AssignmentExpression& parseTree) const;
  evaluation::ParameterValue& BuildAtomicValueFromParseTree(
    const axis::services::language::parsing::SymbolTerminal& parseTree) const;					
  evaluation::ParameterValue& BuildIdFromParseTree(
    const axis::services::language::parsing::IdTerminal& parseTree) const;
  evaluation::ParameterValue& BuildNumberFromParseTree(
    const axis::services::language::parsing::NumberTerminal& parseTree) const;
  evaluation::ParameterValue& BuildStringFromParseTree(
    const axis::services::language::parsing::StringTerminal& parseTree) const;
  void StoreParameterList(evaluation::ParameterList& result);
  void ClearParameterListResult(void);

  evaluation::ParameterList *_paramListResult;
  axis::String _blockName;
  axis::services::language::primitives::OrExpressionParser *_enumParam;
  axis::services::language::primitives::EnumerationParser *_paramList;
  axis::services::language::primitives::AssignmentParser *_assign;
  axis::services::language::primitives::OrExpressionParser *_value;
  axis::services::language::primitives::GeneralExpressionParser *_array;
  axis::services::language::primitives::OrExpressionParser *_arrayVal;
  axis::services::language::primitives::OrExpressionParser *_arrayContents;
  axis::services::language::primitives::EnumerationParser *_arrayEnum;
  axis::services::language::primitives::GeneralExpressionParser *_blockDeclaration;
  axis::services::language::primitives::GeneralExpressionParser *_blockHeaderWithParams;
};			

} } } } // namespace axis::services::language::syntax

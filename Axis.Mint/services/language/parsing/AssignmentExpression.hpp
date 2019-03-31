#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ExpressionNode.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API AssignmentExpression : public ExpressionNode
{
public:
	AssignmentExpression(ParseTreeNode& id, ParseTreeNode& rhs);

	virtual bool IsAssignment(void) const;
	virtual bool IsRhs(void) const;
	virtual bool IsEnumeration(void) const;

	const ParseTreeNode& GetLhs(void) const;
	const ParseTreeNode& GetRhs(void) const;

	virtual ParseTreeNode& Clone(void) const;

	virtual axis::String ToString( void ) const;
	virtual axis::String ToExpressionString(void) const;
private:
	ParseTreeNode *_rhs;
};			

} } } } // namespace axis::services::language::parsing

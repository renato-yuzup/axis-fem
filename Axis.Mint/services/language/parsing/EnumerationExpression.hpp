#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ExpressionNode.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API EnumerationExpression : public ExpressionNode
{
public:
	EnumerationExpression(void);

	virtual bool IsAssignment(void) const;
	virtual bool IsRhs(void) const;
	virtual bool IsEnumeration(void) const;

	virtual axis::String ToString( void ) const;
	virtual axis::String ToExpressionString(void) const;

	virtual ParseTreeNode& Clone(void) const;
};			

} } } } // namespace axis::services::language::parsing

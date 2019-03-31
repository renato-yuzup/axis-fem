#include "RhsExpression.hpp"

namespace aslp = axis::services::language::parsing;

aslp::RhsExpression::RhsExpression( void )
{
	// nothing to do
}

aslp::ParseTreeNode& aslp::RhsExpression::Clone( void ) const
{
	RhsExpression &clone = *new RhsExpression();
	// copy children
	const ParseTreeNode *node = GetFirstChild();
	while (node != NULL)
	{
		clone.AddChild(node->Clone());
		node = node->GetNextSibling();
	}
	return clone;
}

bool aslp::RhsExpression::IsAssignment( void ) const
{
	return false;
}

bool aslp::RhsExpression::IsRhs( void ) const
{
	return true;
}

bool aslp::RhsExpression::IsEnumeration( void ) const
{
	return false;
}
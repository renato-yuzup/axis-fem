#include "EnumerationExpression.hpp"

namespace aslp = axis::services::language::parsing;

aslp::EnumerationExpression::EnumerationExpression( void )
{
	// nothing to do
}

bool aslp::EnumerationExpression::IsAssignment( void ) const
{
	return false;
}

bool aslp::EnumerationExpression::IsRhs( void ) const
{
	return false;
}

bool aslp::EnumerationExpression::IsEnumeration( void ) const
{
	return true;
}

aslp::ParseTreeNode& aslp::EnumerationExpression::Clone( void ) const
{
	EnumerationExpression *clone = new EnumerationExpression();
	const ParseTreeNode *node = GetFirstChild();
	while (node != NULL)
	{
		clone->AddChild(node->Clone());
		node = node->GetNextSibling();
	}
	return *clone;
}

axis::String aslp::EnumerationExpression::ToString( void ) const
{
	axis::String str;
	const ParseTreeNode *node = GetFirstChild();
	while(node != NULL)
	{
		str.append(node->ToString());
		node = node->GetNextSibling();
		if (node != NULL)
		{
			str.append(_T(","));
		}
	}
	return str;
}

axis::String aslp::EnumerationExpression::ToExpressionString( void ) const
{
	axis::String str;
	const ParseTreeNode *node = GetFirstChild();
	while(node != NULL)
	{
		str.append(node->ToExpressionString());
		node = node->GetNextSibling();
		if (node != NULL)
		{
			str.append(_T(","));
		}
	}
	return str;
}
#include "AssignmentExpression.hpp"
#include "foundation/ArgumentException.hpp"

namespace aslp = axis::services::language::parsing;

aslp::AssignmentExpression::AssignmentExpression( ParseTreeNode& id, ParseTreeNode& rhs )
{
	if (!(id.GetPreviousSibling() == NULL && id.GetNextSibling() == NULL))
	{
		throw axis::foundation::ArgumentException();
	}
	if (!(rhs.GetPreviousSibling() == NULL && id.GetNextSibling() == NULL))
	{
		throw axis::foundation::ArgumentException();
	}
	AddChild(id);
	AddChild(rhs);
	_rhs = &rhs;
}

bool aslp::AssignmentExpression::IsAssignment( void ) const
{
	return true;
}

bool aslp::AssignmentExpression::IsRhs( void ) const
{
	return false;
}

bool aslp::AssignmentExpression::IsEnumeration( void ) const
{
	return false;
}

const aslp::ParseTreeNode& aslp::AssignmentExpression::GetLhs( void ) const
{
	return *GetFirstChild();
}

const aslp::ParseTreeNode& aslp::AssignmentExpression::GetRhs( void ) const
{
	return *_rhs;
}

aslp::ParseTreeNode& aslp::AssignmentExpression::Clone( void ) const
{
	return *new AssignmentExpression(GetLhs().Clone(), GetRhs().Clone());
}

axis::String aslp::AssignmentExpression::ToString( void ) const
{
	axis::String str;
	str.append(GetFirstChild()->ToString())
	   .append(_T("="))
	   .append(GetRhs().ToString());
	return str;
}

axis::String aslp::AssignmentExpression::ToExpressionString( void ) const
{
	axis::String str;
	str.append(GetFirstChild()->ToExpressionString())
		.append(_T("="))
		.append(GetRhs().ToExpressionString());
	return str;
}

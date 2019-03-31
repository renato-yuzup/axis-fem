#include "ExpressionNode.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/ApplicationErrorException.hpp"

axis::services::language::parsing::ExpressionNode::ExpressionNode( void )
{
	InitMembers();
}

void axis::services::language::parsing::ExpressionNode::InitMembers( void )
{
	_childrenCount = 0;
	_firstChild = NULL;
	_lastChild = NULL;
}

void axis::services::language::parsing::ExpressionNode::AddChild( ParseTreeNode& node )
{
	int newChildrenCount = 0;
	ParseTreeNode *newChild = &node;
	ParseTreeNode *lastChild;

	if (node.GetPreviousSibling() != NULL)
	{
		throw axis::foundation::ArgumentException();
	}

	while (newChild != NULL)
	{
		newChild->SetParent(*this);
		lastChild = newChild;
		newChild = newChild->GetNextSibling();
		++newChildrenCount;
	}
	if (_firstChild == NULL)
	{
		_firstChild = &node;
		_firstChild->NotifyUse();
	}
	if (_lastChild != NULL)
	{
		_lastChild->SetNextSibling(node);
	}
	_childrenCount += newChildrenCount;
	_lastChild = lastChild;
}

axis::services::language::parsing::ExpressionNode::~ExpressionNode( void )
{
	// notify first child node that we are no longer referencing it
	if (_firstChild != NULL) _firstChild->NotifyDestroy();
}

const axis::services::language::parsing::ParseTreeNode *axis::services::language::parsing::ExpressionNode::GetFirstChild( void ) const
{
	return _firstChild;
}

const axis::services::language::parsing::ParseTreeNode *axis::services::language::parsing::ExpressionNode::GetLastChild( void ) const
{
	return _lastChild;
}

int axis::services::language::parsing::ExpressionNode::GetChildCount( void ) const
{
	return _childrenCount;
}

bool axis::services::language::parsing::ExpressionNode::IsEmpty( void ) const
{
	return _childrenCount == 0;
}

axis::String axis::services::language::parsing::ExpressionNode::ToString( void ) const
{
	axis::String str;
	ParseTreeNode *node = _firstChild;
	while(node != NULL)
	{
		str.append(node->ToString());
		node = node->GetNextSibling();
	}
	return str;
}

bool axis::services::language::parsing::ExpressionNode::IsTerminal( void ) const
{
	return false;
}

axis::String axis::services::language::parsing::ExpressionNode::ToExpressionString( void ) const
{
	axis::String str;
	ParseTreeNode *node = _firstChild;
	while(node != NULL)
	{
		axis::String nextExpr = node->BuildExpressionString();
		if (!nextExpr.empty() && !str.empty())
		{
			if (CheckCharType(str[str.length() - 1]) == CheckCharType(nextExpr[0]))
			{
				str.append(_T(" "));
			}
		}
		str.append(nextExpr);
		node = node->GetNextSibling();
	}
	return str;
}

axis::services::language::parsing::ExpressionNode::CharType axis::services::language::parsing::ExpressionNode::CheckCharType( axis::String::value_type c ) const
{
	if ((c >= (axis::String::value_type)'A' && c <= (axis::String::value_type)'Z') ||
		(c >= (axis::String::value_type)'a' && c <= (axis::String::value_type)'z') ||
		(c >= (axis::String::value_type)'0' && c <= (axis::String::value_type)'9') ||
		(c == (axis::String::value_type)'_'))
	{	
		return IdentifierDeclarationType;
	}
	return NonIdentifierDeclarationType;
}
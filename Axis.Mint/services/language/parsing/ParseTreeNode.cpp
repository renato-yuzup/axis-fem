#include "ParseTreeNode.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aslp = axis::services::language::parsing;

aslp::ParseTreeNode::ParseTreeNode( void )
{
	InitMembers();
}

aslp::ParseTreeNode::ParseTreeNode( ParseTreeNode &parent )
{
	InitMembers();
	_parent = &parent;
}

aslp::ParseTreeNode::ParseTreeNode( ParseTreeNode &parent, ParseTreeNode &previousSibling )
{
	InitMembers();
	_parent = &parent;
	_previouSibling = &previousSibling;
}

aslp::ParseTreeNode::~ParseTreeNode( void )
{
	// tell sibling node that we are no longer referencing it
	if (_nextSibling != NULL) _nextSibling->NotifyDestroy();
}

void aslp::ParseTreeNode::InitMembers( void )
{	// clear variables
	_parent = NULL;
	_previouSibling = NULL;
	_nextSibling = NULL;
	_useCount = 0;
}

void aslp::ParseTreeNode::SetParent( ParseTreeNode& parent )
{
	if (_parent != NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	_parent = &parent;
}

aslp::ParseTreeNode *aslp::ParseTreeNode::GetParent( void ) const
{
	return _parent;
}

aslp::ParseTreeNode *aslp::ParseTreeNode::GetPreviousSibling( void ) const
{
	return _previouSibling;
}

aslp::ParseTreeNode *aslp::ParseTreeNode::GetNextSibling( void ) const
{
	return _nextSibling;
}

bool aslp::ParseTreeNode::IsRoot( void ) const
{ 
	return _parent == NULL;
}

aslp::ParseTreeNode& aslp::ParseTreeNode::SetNextSibling( ParseTreeNode& nextSibling )
{
	if (_nextSibling != NULL || nextSibling.GetPreviousSibling() != NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	_nextSibling = &nextSibling;
	nextSibling.NotifyUse();

	_nextSibling->_previouSibling = this;

	return *_nextSibling;
}

void aslp::ParseTreeNode::NotifyUse( void )
{
	++_useCount;
}

void aslp::ParseTreeNode::NotifyDestroy( void )
{
	--_useCount;
	if (_useCount <= 0)
	{	// no other object hold reference to this object; delete it
		delete this;
	}
}

axis::String aslp::ParseTreeNode::BuildExpressionString( void ) const
{
	axis::String expr = ToExpressionString();
 	return expr;
}

axis::String aslp::ParseTreeNode::ToExpressionString( void ) const
{
	/* default implementation is to use the same algorithm as ToString() */
	return ToString();
}

aslp::ParseTreeNode::CharType aslp::ParseTreeNode::CheckCharType( axis::String::value_type c ) const
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
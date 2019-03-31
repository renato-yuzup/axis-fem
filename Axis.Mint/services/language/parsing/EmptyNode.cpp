#include "EmptyNode.hpp"

namespace aslp = axis::services::language::parsing;

aslp::EmptyNode::EmptyNode( void )
{
	// nothing to do here
}

bool aslp::EmptyNode::IsId( void ) const
{
	return false;
}

bool aslp::EmptyNode::IsNumber( void ) const
{
	return false;
}

bool aslp::EmptyNode::IsString( void ) const
{
	return false;
}

bool aslp::EmptyNode::IsReservedWord( void ) const
{
	return false;
}

bool aslp::EmptyNode::IsOperator( void ) const
{
	return false;
}

aslp::ParseTreeNode& aslp::EmptyNode::Clone( void ) const
{
	return *new EmptyNode();
}

axis::String aslp::EmptyNode::ToString( void ) const
{
	return _T("");
}

bool aslp::EmptyNode::IsEmpty( void ) const
{
	return true;
}

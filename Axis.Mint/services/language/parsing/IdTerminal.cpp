#include "IdTerminal.hpp"

namespace aslp = axis::services::language::parsing;

aslp::IdTerminal::IdTerminal( const axis::String& id ) : _id(id)
{
	// nothing to do
}

bool aslp::IdTerminal::IsId( void ) const
{
	return true;	
}

bool aslp::IdTerminal::IsNumber( void ) const
{
	return false;
}

bool aslp::IdTerminal::IsString( void ) const
{
	return false;
}

bool aslp::IdTerminal::IsReservedWord( void ) const
{
	return false;
}

bool aslp::IdTerminal::IsOperator( void ) const
{
	return false;
}

aslp::ParseTreeNode& aslp::IdTerminal::Clone( void ) const
{
	return *new IdTerminal(_id);
}

axis::String aslp::IdTerminal::ToString( void ) const
{
	return _id;
}

axis::String aslp::IdTerminal::GetId( void ) const
{
	return _id;
}
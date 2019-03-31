#include "ReservedWordTerminal.hpp"

namespace aslp = axis::services::language::parsing;

aslp::ReservedWordTerminal::ReservedWordTerminal( const axis::String& name, int associatedValue ) :
_name(name), _associatedValue(associatedValue)
{
	 /* nothing to do here */
}

bool aslp::ReservedWordTerminal::IsId( void ) const
{
	return false;
}

bool aslp::ReservedWordTerminal::IsNumber( void ) const
{
	return false;
}

bool aslp::ReservedWordTerminal::IsString( void ) const
{
	return false;
}

bool aslp::ReservedWordTerminal::IsReservedWord( void ) const
{
	return true;
}

bool aslp::ReservedWordTerminal::IsOperator( void ) const
{
	return false;
}

aslp::ParseTreeNode& aslp::ReservedWordTerminal::Clone( void ) const
{
	return *new ReservedWordTerminal(_name, _associatedValue);
}

axis::String aslp::ReservedWordTerminal::ToString( void ) const
{
	return _name;
}

int aslp::ReservedWordTerminal::GetValue( void ) const
{
	return _associatedValue;
}
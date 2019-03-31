#include "OperatorTerminal.hpp"

namespace aslp = axis::services::language::parsing;

aslp::OperatorTerminal::OperatorTerminal( const axis::String& name, int associatedValue, 
                                          int precedence, int associativity ) :
_associatedValue(associatedValue), _name(name), _precedence(precedence), _associativity(associativity)
{
	/* nothing to do here */
}

bool aslp::OperatorTerminal::IsId( void ) const
{
	return false;
}

bool aslp::OperatorTerminal::IsNumber( void ) const
{
	return false;
}

bool aslp::OperatorTerminal::IsString( void ) const
{
	return false;
}

bool aslp::OperatorTerminal::IsReservedWord( void ) const
{
	return false;
}

bool aslp::OperatorTerminal::IsOperator( void ) const
{
	return true;
}

aslp::ParseTreeNode& aslp::OperatorTerminal::Clone( void ) const
{
	return *new OperatorTerminal(_name, _associatedValue, _precedence, _associativity);
}

axis::String aslp::OperatorTerminal::ToString( void ) const
{
	return _name;
}

int aslp::OperatorTerminal::GetValue( void ) const
{
	return _associatedValue;
}

int aslp::OperatorTerminal::GetPrecedence( void ) const
{
	return _precedence;
}

int aslp::OperatorTerminal::GetAssociativity( void ) const
{
	return _associativity;
}
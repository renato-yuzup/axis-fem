#include "NumberTerminal.hpp"

namespace aslp = axis::services::language::parsing;

aslp::NumberTerminal::NumberTerminal( long num, const axis::String& stringRepresentation ) :
_decNum((double)num), _intNum(num), _isInteger(true), _strRepr(stringRepresentation)
{
	// nothing more to do
}

aslp::NumberTerminal::NumberTerminal( double num, const axis::String& stringRepresentation ) :
_decNum(num), _intNum((long)num), _isInteger(false), _strRepr(stringRepresentation)
{
	// nothing more to do
}

bool aslp::NumberTerminal::IsId( void ) const
{
	return false;
}

bool aslp::NumberTerminal::IsNumber( void ) const
{
	return true;
}

bool aslp::NumberTerminal::IsString( void ) const
{
	return false;
}

bool aslp::NumberTerminal::IsReservedWord( void ) const
{
	return false;
}

bool aslp::NumberTerminal::IsOperator( void ) const
{
	return false;
}

aslp::ParseTreeNode& aslp::NumberTerminal::Clone( void ) const
{
	if (_isInteger)
	{
		return *new NumberTerminal(_intNum, _strRepr);
	}
	else
	{
		return *new NumberTerminal(_decNum, _strRepr);
	}
}

axis::String aslp::NumberTerminal::ToString( void ) const
{
	return _strRepr;
}

bool aslp::NumberTerminal::IsInteger( void ) const
{
	return _isInteger;
}

long aslp::NumberTerminal::GetInteger( void ) const
{
	return _intNum;
}

double aslp::NumberTerminal::GetDouble( void ) const
{
	return _decNum;
}
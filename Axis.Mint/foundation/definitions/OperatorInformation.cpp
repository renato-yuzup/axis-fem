#include "OperatorInformation.hpp"

namespace afd = axis::foundation::definitions;

afd::OperatorInformation::OperatorInformation( const axis::String::char_type * syntax, int precedence, 
                                               OperatorAssociativity associativity, TokenType type ) :
_syntax(syntax)
{
	_precedence = precedence;
	_associativity = associativity;
	_type = type;
}

int afd::OperatorInformation::GetPrecedence( void ) const
{
	return _precedence;
}

afd::OperatorAssociativity afd::OperatorInformation::GetAssociativity( void ) const
{
	return _associativity;
}

axis::String afd::OperatorInformation::GetSyntax( void ) const
{
	return _syntax;
}

bool afd::OperatorInformation::IsLeftAssociative( void ) const
{
	return GetAssociativity() == kLeftAssociativity;
}

bool afd::OperatorInformation::IsRightAssociative( void ) const
{
	return GetAssociativity() == kRightAssociativity;
}

afd::TokenType afd::OperatorInformation::GetType( void ) const
{
	return _type;
}
#pragma once
#include "OperatorAssociativity.hpp"
#include "AxisString.hpp"
#include "TokenType.hpp"
#include "foundation/Axis.Mint.hpp"

namespace axis { namespace foundation { namespace definitions {

class AXISMINT_API OperatorInformation
{
private:
	int _precedence;
	OperatorAssociativity _associativity;
	const axis::String::char_type * _syntax;
	TokenType _type;
public:
	OperatorInformation(const axis::String::char_type * syntax, int precedence, OperatorAssociativity associativity, TokenType type);
	int GetPrecedence(void) const;
	OperatorAssociativity GetAssociativity(void) const;
	axis::String GetSyntax(void) const;
	TokenType GetType(void) const;

	bool IsLeftAssociative(void) const;
	bool IsRightAssociative(void) const;
};	

} } } // namespace axis::foundation::definitions

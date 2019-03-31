#include "Symbol.hpp"

namespace aapp = axis::application::parsing::preprocessing;
namespace afd = axis::foundation::definitions;

aapp::Symbol::Symbol( afd::TokenType type, const axis::String& name )
{
	Type = type;
	Name = name;
	Precedence = 0;
	Associativity = afd::kLeftAssociativity;
}

aapp::Symbol::Symbol( afd::TokenType type, const axis::String& name, const axis::String& value )
{
	Type = type;
	Name = name;
	Value = value;
	Precedence = 0;
	Associativity = afd::kLeftAssociativity;
}

aapp::Symbol::Symbol( afd::TokenType type, const axis::String& name, int precedence )
{
	Type = type;
	Name = name;
	Precedence = precedence;
	Associativity = afd::kLeftAssociativity;
}

aapp::Symbol::Symbol( afd::TokenType type, const axis::String& name, const axis::String& value, 
                      int precedence )
{
	Type = type;
	Name = name;
	Value = value;
	Precedence = precedence;
	Associativity = afd::kLeftAssociativity;
}

aapp::Symbol::Symbol( afd::TokenType type, const axis::String& name, int precedence, 
                      afd::OperatorAssociativity associativity )
{
	Type = type;
	Name = name;
	Precedence = precedence;
	Associativity = associativity;
}

aapp::Symbol::Symbol( afd::TokenType type, const axis::String& name, const axis::String& value, 
                      int precedence, afd::OperatorAssociativity associativity )
{
	Type = type;
	Name = name;
	Value = value;
	Precedence = precedence;
	Associativity = associativity;
}

aapp::Symbol::~Symbol(void)
{
	// no op
}
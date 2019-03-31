#pragma once
#include "AxisString.hpp"
#include "foundation/definitions/OperatorAssociativity.hpp"
#include "foundation/definitions/TokenType.hpp"

namespace axis { namespace application { namespace parsing { namespace preprocessing {

struct Symbol
{
public:
  Symbol(axis::foundation::definitions::TokenType type, const axis::String& name);
  Symbol(axis::foundation::definitions::TokenType type, const axis::String& name, int precedence);
  Symbol(axis::foundation::definitions::TokenType type, const axis::String& name, int precedence, 
    axis::foundation::definitions::OperatorAssociativity associativity);
  Symbol(axis::foundation::definitions::TokenType type, const axis::String& name, 
    const axis::String& value);
  Symbol(axis::foundation::definitions::TokenType type, const axis::String& name, 
    const axis::String& value, int precedence);
  Symbol(axis::foundation::definitions::TokenType type, const axis::String& name, 
    const axis::String& value, int precedence, 
    axis::foundation::definitions::OperatorAssociativity associativity);
  ~Symbol(void);

  axis::foundation::definitions::TokenType Type;
	axis::String Name;
	axis::String Value;
	int Precedence;
	axis::foundation::definitions::OperatorAssociativity Associativity;
};

} } } } // namespace axis::application::parsing::preprocessing

#include "SymbolTerminal.hpp"

namespace aslp = axis::services::language::parsing;

aslp::SymbolTerminal::~SymbolTerminal( void )
{
	// nothing to do
}

bool aslp::SymbolTerminal::IsTerminal( void ) const
{
	return true;
}

bool aslp::SymbolTerminal::IsEmpty( void ) const
{
	return false;
}
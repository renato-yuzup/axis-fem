#include "SymbolTable.hpp"
#include "foundation/ArgumentException.hpp"

namespace aapp = axis::application::parsing::preprocessing;

aapp::SymbolTable::SymbolTable(void)
{
	// no op
}

aapp::SymbolTable::~SymbolTable(void)
{
	// erases symbol table and call destructors of every every contained element
	_table.clear();
}

void aapp::SymbolTable::AddSymbol( Symbol& symbol )
{
	if (IsDefined(symbol.Name))
	{
		throw axis::foundation::ArgumentException();
	}
	_table[symbol.Name] = &symbol;
}

void aapp::SymbolTable::ClearTable( void )
{
	_table.clear();
}

bool aapp::SymbolTable::IsDefined( const axis::String& name ) const
{
	symbol_table::const_iterator it = _table.find(name);
	return (it != _table.end());
}

const axis::String& aapp::SymbolTable::GetValue( const axis::String& name ) const
{
	symbol_table::const_iterator it = _table.find(name);
	if (it == _table.end())
	{	// element not found
		throw axis::foundation::ArgumentException();
	}
	return (it->second)->Value;
}

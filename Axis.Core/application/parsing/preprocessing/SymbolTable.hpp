#pragma once

#include "AxisString.hpp"
#include "Symbol.hpp"
#include <map>

namespace axis { namespace application { namespace parsing { namespace preprocessing {

class SymbolTable
{
public:
	SymbolTable(void);
	~SymbolTable(void);
	void AddSymbol(Symbol& symbol);
	void ClearTable(void);
	bool IsDefined(const axis::String& name) const;
	const axis::String& GetValue(const axis::String& name) const;
private:
  typedef std::map<axis::String, Symbol *, axis::StringCompareLessThan> symbol_table;
  symbol_table _table;
};

} } } } // namespace axis::application::parsing::preprocessing

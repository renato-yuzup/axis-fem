#pragma once
#include "foundation/Axis.Mint.hpp"
#include "SymbolTerminal.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API IdTerminal : public SymbolTerminal
{
public:
	IdTerminal(const axis::String& id);
				
	virtual bool IsId(void) const;
	virtual bool IsNumber(void) const;
	virtual bool IsString(void) const;
	virtual bool IsReservedWord(void) const;
	virtual bool IsOperator(void) const;

	virtual ParseTreeNode& Clone(void) const;
	virtual axis::String ToString(void) const;

	axis::String GetId(void) const;
private:
	const axis::String _id;
};					

} } } } // namespace axis::services::language::parsing

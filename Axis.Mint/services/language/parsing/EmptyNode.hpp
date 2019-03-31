#pragma once
#include "foundation/Axis.Mint.hpp"
#include "SymbolTerminal.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API EmptyNode : public SymbolTerminal
{
public:
	EmptyNode(void);

	virtual bool IsId(void) const;
	virtual bool IsNumber(void) const;
	virtual bool IsString(void) const;
	virtual bool IsReservedWord(void) const;
	virtual bool IsOperator(void) const;
	virtual bool IsEmpty(void) const;

	virtual ParseTreeNode& Clone(void) const;
	virtual axis::String ToString(void) const;
};					

} } } } // namespace axis::services::language::parsing

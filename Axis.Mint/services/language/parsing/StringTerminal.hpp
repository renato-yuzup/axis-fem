#pragma once
#include "foundation/Axis.Mint.hpp"
#include "SymbolTerminal.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API StringTerminal : public SymbolTerminal
{
private:
	const axis::String _value;
	axis::String InsertEscapeChars(const axis::String& s) const;
public:
	StringTerminal(const axis::String& value);

	virtual bool IsId(void) const;
	virtual bool IsNumber(void) const;
	virtual bool IsString(void) const;
	virtual bool IsReservedWord(void) const;
	virtual bool IsOperator(void) const;

	virtual ParseTreeNode& Clone(void) const;
	virtual axis::String ToString(void) const;
	virtual axis::String ToExpressionString(void) const;
};					

} } } } // namespace axis::services::language::parsing

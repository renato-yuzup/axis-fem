#pragma once
#include "foundation/Axis.Mint.hpp"
#include "SymbolTerminal.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API OperatorTerminal : public SymbolTerminal
{
public:
	OperatorTerminal(const axis::String& name, int associatedValue, int precedence, int associativity);

	virtual bool IsId(void) const;
	virtual bool IsNumber(void) const;
	virtual bool IsString(void) const;
	virtual bool IsReservedWord(void) const;
	virtual bool IsOperator(void) const;

	virtual ParseTreeNode& Clone(void) const;
	virtual axis::String ToString(void) const;

	int GetValue(void) const;
	int GetPrecedence(void) const;
	int GetAssociativity(void) const;
private:
	const axis::String _name;
	const int _associatedValue;
	const int _precedence;
	const int _associativity;
};				

} } } } // namespace axis::services::language::parsing

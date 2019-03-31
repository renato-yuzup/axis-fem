#pragma once
#include "foundation/Axis.Mint.hpp"
#include "SymbolTerminal.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API NumberTerminal : public SymbolTerminal
{
public:
	NumberTerminal(long num, const axis::String& stringRepresentation);
	NumberTerminal(double num, const axis::String& stringRepresentation);

	virtual bool IsId(void) const;
	virtual bool IsNumber(void) const;
	virtual bool IsString(void) const;
	virtual bool IsReservedWord(void) const;
	virtual bool IsOperator(void) const;

	virtual ParseTreeNode& Clone(void) const;
	virtual axis::String ToString(void) const;

	bool IsInteger(void) const;
	long GetInteger(void) const;
	double GetDouble(void) const;
private:
	const axis::String _strRepr;
	const long _intNum;
	const double _decNum;
	const bool _isInteger;
};		

} } } } // namespace axis::services::language::parsing


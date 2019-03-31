#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParseTreeNode.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API SymbolTerminal : public ParseTreeNode
{
public:
	virtual ~SymbolTerminal(void);
	virtual bool IsTerminal( void ) const;
	virtual bool IsId(void) const = 0;
	virtual bool IsNumber(void) const = 0;
	virtual bool IsString(void) const = 0;
	virtual bool IsReservedWord(void) const = 0;
	virtual bool IsOperator(void) const = 0;
	virtual bool IsEmpty(void) const;
};			

} } } } // namespace axis::services::language::parsing


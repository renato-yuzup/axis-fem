#pragma once
#include "ExpressionParser.hpp"
#include "foundation/Axis.Mint.hpp"
#include "ExpressionList.hpp"
#include "Parser.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API CompositeParser : public ExpressionParser
{
public:
	CompositeParser(void);
	virtual ~CompositeParser(void);
	virtual CompositeParser& Add( const ExpressionParser& expression );
	virtual CompositeParser& Add( const Parser& parser );
	virtual void Remove( const ExpressionParser& expression );
	CompositeParser& operator <<(ExpressionParser& expression);
	CompositeParser& operator <<(const Parser& parser);
protected:
	ExpressionList _components;
private:
	ExpressionList _ownedExpressionObjs;
};						

} } } } // namespace axis::services::language::primitives

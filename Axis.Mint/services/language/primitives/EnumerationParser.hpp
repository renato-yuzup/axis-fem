#pragma once
#include "CompositeParser.hpp"
#include "foundation/Axis.Mint.hpp"
#include "AtomicExpressionParser.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API EnumerationParser : public CompositeParser
{
private:
	AtomicExpressionParser _separatorParser;
	const ExpressionParser *_enumeratedExpression;
	const bool _ownedExpressionParser;
	const bool _isRequired;
protected:
	virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
public:
	EnumerationParser(const ExpressionParser& enumeratedExpression, bool isRequired = false);
	EnumerationParser(const Parser& enumeratedExpression, bool isRequired = false);
	~EnumerationParser(void);

	// overload these functions so that we won't allow direct inclusion of expressions
	virtual CompositeParser& Add( ExpressionParser& expression );
	virtual CompositeParser& Add( const Parser& parser );
	virtual void Remove( const ExpressionParser& expression );
};			

} } } } // namespace axis::services::language::primitives

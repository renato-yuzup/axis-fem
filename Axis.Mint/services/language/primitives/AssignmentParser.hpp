#pragma once
#include "foundation/Axis.Mint.hpp"
#include "CompositeParser.hpp"
#include "GeneralExpressionParser.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API AssignmentParser : public CompositeParser
{
public:
	AssignmentParser(void);
	~AssignmentParser(void);
	void SetRhsExpression(const ExpressionParser& expression);
	// overload these functions so that we won't allow direct inclusion of expressions
	virtual CompositeParser& Add( ExpressionParser& expression );
	virtual CompositeParser& Add( const Parser& parser );
	virtual void Remove( const ExpressionParser& expression );
private:
  virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;

	GeneralExpressionParser _parser;
	const ExpressionParser *_rhs;
	bool _isNullExpression;
};			

} } } } // namespace axis::services::language::primitives

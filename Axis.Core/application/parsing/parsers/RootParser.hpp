#pragma once
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class RootParser : public axis::application::parsing::parsers::BlockParser
{
public:
	RootParser(axis::application::factories::parsers::BlockProvider& factory);
	virtual ~RootParser(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext(
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
private:
	void InitGrammar(void);

	// our grammar
	axis::services::language::primitives::OrExpressionParser _titleExpression;
	axis::services::language::primitives::GeneralExpressionParser _expression1;
	axis::services::language::primitives::GeneralExpressionParser _expression2;
	axis::services::language::primitives::OrExpressionParser _titleSeparator;

	axis::application::factories::parsers::BlockProvider& _parent;
};	

} } } } // namespace axis::application::parsing::parsers

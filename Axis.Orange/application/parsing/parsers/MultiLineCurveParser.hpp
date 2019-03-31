#pragma once
#include "application/parsing/parsers/BlockParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include <list>

namespace axis { namespace application { namespace parsing { namespace parsers {

class MultiLineCurveParser : public axis::application::parsing::parsers::BlockParser
{
public:
	MultiLineCurveParser(const axis::String& curveId);
	~MultiLineCurveParser(void);
	virtual void DoCloseContext( void );
	virtual void DoStartContext( void );
	virtual BlockParser& GetNestedContext( const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
private:
  struct Point
  {
  public:
    double x;
    double y;
  };
	typedef std::list<Point> point_list;

	void InitGrammar(void);
	Point ParsePointExpression( const axis::services::language::parsing::ParseTreeNode& parseTree );

  point_list _points;
	axis::String _curveId;
	bool _onErrorRecovery;
	bool _ignoreCurve;

	// our grammar rules	
	axis::services::language::primitives::OrExpressionParser _acceptedExpressions;
	axis::services::language::primitives::OrExpressionParser _valueSeparator;
	axis::services::language::primitives::GeneralExpressionParser _nonBlankValueSeparator;
	axis::services::language::primitives::OrExpressionParser _acceptedValueSeparator;
	axis::services::language::primitives::GeneralExpressionParser _groupedExpression1;
	axis::services::language::primitives::GeneralExpressionParser _groupedExpression2;
	axis::services::language::primitives::GeneralExpressionParser _groupedExpression3;
	axis::services::language::primitives::GeneralExpressionParser _ungroupedExpression;
};

} } } } // namespace axis::application::parsing::parsers

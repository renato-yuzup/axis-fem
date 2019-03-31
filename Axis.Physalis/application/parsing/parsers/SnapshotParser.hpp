#pragma once
#include "application/parsing/parsers/BlockParser.hpp"
#include <list>
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class SnapshotParser : public axis::application::parsing::parsers::BlockParser
{
public:
	SnapshotParser(bool ignoreSnapshotDeclarations);
	~SnapshotParser(void);
	virtual void DoCloseContext( void );
	virtual void DoStartContext( void );
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
	void ParseAbsoluteStatement( const axis::services::language::parsing::ParseTreeNode& parseTree );
	void ParseRelativeStatement( const axis::services::language::parsing::ParseTreeNode& parseTree );
	void ParseRangeStatement( const axis::services::language::parsing::ParseTreeNode& parseTree );
	void ParseSplitStatement( const axis::services::language::parsing::ParseTreeNode& parseTree );
	void ParseSnapshotStatement( const axis::services::language::parsing::ParseTreeNode& parseTree );
	bool AddSnapshotMark( real timeValue, bool isRelative );
private:
  void InitGrammar(void);

	bool _ignoreSnapshotDeclarations;
	bool _dirtySnapshotGroup;
	bool _canAddMoreSnapshots;
	real _lastSnapshotTime;
	std::list<real> _marks;
	axis::services::language::primitives::OrExpressionParser _acceptedStatements;
	axis::services::language::primitives::GeneralExpressionParser _absoluteMarkStatement;
	axis::services::language::primitives::GeneralExpressionParser _relativeMarkStatement;
	axis::services::language::primitives::GeneralExpressionParser _rangeMarkStatement;
	axis::services::language::primitives::GeneralExpressionParser _splitStatement;
};	

} } } } // namespace axis::application::parsing::parsers

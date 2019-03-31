#pragma once
#include "application/parsing/parsers/BlockParser.hpp"

class MockBlockParser : public axis::application::parsing::parsers::BlockParser
{
private:
	bool _simulateBadParser;
	bool _failOnStart;
	bool _failOnClose;
	bool _failOnRead;

	axis::String _lastLineRead;

	axis::String _expectedContextName;
	axis::services::language::syntax::evaluation::ParameterList *_paramListReceived;
	BlockParser *_parserToReturn;
public:
	MockBlockParser(bool simulateBadParser = false);
	MockBlockParser(bool simulateBadParser, bool failOnStartContext, bool failOnCloseContext, bool failOnRead);
	~MockBlockParser(void);

	virtual void DoCloseContext( void );

	virtual void DoStartContext( void );

	virtual BlockParser& GetNestedContext( const axis::String& contextName, const axis::services::language::syntax::evaluation::ParameterList& paramList );

	virtual axis::services::language::parsing::ParseResult Parse(const axis::services::language::iterators::InputIterator& begin, const axis::services::language::iterators::InputIterator& end);

	axis::String GetLastLineRead(void) const;

	void SetExpectedNestedBlock(const axis::String& contextName, BlockParser& parserToReturn);
	void ClearExpectedNestedBlock(void);
	const axis::services::language::syntax::evaluation::ParameterList& GetParamListReceived(void) const;
};


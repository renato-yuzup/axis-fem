#if defined DEBUG || defined _DEBUG

#include "MockBlockParser.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/NotSupportedException.hpp"

using namespace axis::application::parsing::parsers;

MockBlockParser::MockBlockParser( bool simulateBadParser /*= false*/ )
{
	_simulateBadParser = simulateBadParser;
	_failOnStart = true;
	_failOnClose = true;
	_failOnRead = true;

	_paramListReceived = NULL;
	_parserToReturn = NULL;
}

MockBlockParser::MockBlockParser( bool simulateBadParser, bool failOnStartContext, bool failOnCloseContext, bool failOnRead )
{
	_simulateBadParser = simulateBadParser;
	_failOnStart = failOnStartContext;
	_failOnClose = failOnCloseContext;
	_failOnRead = failOnRead;

	_paramListReceived = NULL;
	_parserToReturn = NULL;
}

MockBlockParser::~MockBlockParser(void)
{
	if (_paramListReceived != NULL)
	{
		delete _paramListReceived;
	}
}

void MockBlockParser::DoCloseContext( void )
{
	if (_simulateBadParser && _failOnClose)
	{
		throw axis::foundation::OutOfMemoryException();
	}
}

void MockBlockParser::DoStartContext( void )
{
	if (_simulateBadParser && _failOnStart)
	{
		throw axis::foundation::OutOfMemoryException();
	}
}

BlockParser& MockBlockParser::GetNestedContext( const axis::String& contextName, const axis::services::language::syntax::evaluation::ParameterList& paramList )
{
	if (_parserToReturn == NULL)
	{
		throw axis::foundation::NotSupportedException();
	}
	if (_expectedContextName.compare(contextName) == 0)
	{
		if (_paramListReceived != NULL)
		{
			delete _paramListReceived;
		}
		_paramListReceived = &paramList.Clone();
		return *_parserToReturn;
	}
	else
	{
		throw axis::foundation::NotSupportedException();
	}
}

axis::services::language::parsing::ParseResult MockBlockParser::Parse(const axis::services::language::iterators::InputIterator& begin, const axis::services::language::iterators::InputIterator& end)
{
	if (_failOnRead && _simulateBadParser)
	{
		throw axis::foundation::OutOfMemoryException();
	}
	_lastLineRead = begin.ToString(end);
	axis::services::language::parsing::ParseResult result;
	result.SetResult(axis::services::language::parsing::ParseResult::MatchOk);
	result.SetLastReadPosition(end);
	return result;
}

void MockBlockParser::SetExpectedNestedBlock( const axis::String& contextName, BlockParser& parserToReturn )
{
	_expectedContextName = contextName;
	_parserToReturn = &parserToReturn;
}

void MockBlockParser::ClearExpectedNestedBlock( void )
{
	_expectedContextName = _T("");
	_parserToReturn = NULL;
}

axis::String MockBlockParser::GetLastLineRead( void ) const
{
	return _lastLineRead;
}

const axis::services::language::syntax::evaluation::ParameterList& MockBlockParser::GetParamListReceived( void ) const
{
	return *_paramListReceived;
}

#endif
#include "ParseResult.hpp"
#include "foundation/ApplicationErrorException.hpp"
#include "EmptyNode.hpp"

namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;

aslp::ParseResult::ParseResult( Result result, ParseTreeNode& parseTree, 
                                const asli::InputIterator& lastPosition ) 
{
	_result = result;
	_parseTree = &parseTree;
	_parseTree->NotifyUse();
	_lastPosition = NULL;
	_lastPosition = &lastPosition.Clone();
}

aslp::ParseResult::ParseResult( const ParseResult& other ) 
{
	_parseTree = NULL;
	_lastPosition = NULL;
	Copy(other);
}

aslp::ParseResult::ParseResult( void ) 
{
	_result = FailedMatch;
	_parseTree = new EmptyNode();
	_parseTree->NotifyUse();
	_lastPosition = new asli::InputIterator();
}

aslp::ParseResult::~ParseResult( void )
{
	_parseTree->NotifyDestroy();
	delete _lastPosition;
}

bool aslp::ParseResult::IsMatch( void ) const
{
	return _result == MatchOk;
}

aslp::ParseResult::Result aslp::ParseResult::GetResult( void ) const
{
	return _result;
}

aslp::ParseTreeNode& aslp::ParseResult::GetParseTree( void )
{
	return *_parseTree;
}

const aslp::ParseTreeNode& aslp::ParseResult::GetParseTree( void ) const
{
	return *_parseTree;
}

asli::InputIterator aslp::ParseResult::GetLastReadPosition( void ) const
{
	return *_lastPosition;
}

aslp::ParseResult& aslp::ParseResult::operator=( const ParseResult& other )
{
	Copy(other);
	return *this;
}

void aslp::ParseResult::SetResult( const Result result )
{
	_result = result;
}

void aslp::ParseResult::SetLastReadPosition( const asli::InputIterator& it )
{
	*_lastPosition = it;
}

void aslp::ParseResult::ClearParseTree( void )
{
	_parseTree->NotifyDestroy();
	_parseTree = new EmptyNode();
	_parseTree->NotifyUse();
}

void aslp::ParseResult::Copy( const ParseResult& other )
{
	// create our new objects
	//aslp::ParseTreeNode *newTree = NULL;
	asli::InputIterator *newIt = NULL;
	try
	{
		//newTree = &other._parseTree->Clone();
		newIt = &other._lastPosition->Clone();
	}
	catch (...)
	{	// something went wrong, delete objects created and rethrow a new exception
		//if (newTree != NULL) delete newTree;
		if (newIt != NULL) delete newIt;
		throw axis::foundation::ApplicationErrorException();
	}

	//delete old copies (if exist)
	//if (_parseTree != NULL) delete _parseTree;
	if (_lastPosition != NULL) delete _lastPosition;
	if (other._parseTree != _parseTree)
	{
		if (_parseTree != NULL)
		{
			_parseTree->NotifyDestroy();
		}
		_parseTree = other._parseTree;
		_parseTree->NotifyUse();
	}
	 
	// change values
	_result = other._result;
	_lastPosition = newIt;
	//_parseTree = newTree;
}
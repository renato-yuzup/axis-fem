#include "StatementDecoder.hpp"

#include "foundation/StandardTraceHints.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/Level2ParserException.hpp"
#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/MissingDelimiterException.hpp"
#include "foundation/UnexpectedExpressionException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"

#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/grammar_tokens.hpp"

#include "log_messages/parser_messages.hpp"
#include "application/parsing/error_messages.hpp"
#include "services/messaging/ErrorMessage.hpp"

namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aslp = axis::services::language::parsing;
namespace asli = axis::services::language::iterators;
namespace aslf = axis::services::language::factories;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aslpp = axis::services::language::primitives;
namespace asmm = axis::services::messaging;

aapc::StatementDecoder::StatementDecoder(aapc::ParseContext& parseContext) :
_parseContext(parseContext)
{
	_maxBufferLength = 4*1024*1024;
	_currentContext = NULL;
	_errorState = NoError;
	_onErrorRecovery = false;
	_onBlockHeadRead = false;
	_onBlockTailRead = false;
	_onBodyRead = false;
	_wasFinalized = false;
	InitGrammar();
}

aapc::StatementDecoder::~StatementDecoder(void)
{
}

void aapc::StatementDecoder::InitGrammar( void )
{
}

aapc::StatementDecoder::ParsingState aapc::StatementDecoder::TestAndRunForBlockHeader( 
  const asli::InputIterator& begin, const asli::InputIterator& end )
{
	aslp::ParseResult result = _blockHeaderParser(begin, end);
	if (result.IsMatch())
	{	// found a block header
		// clear error condition
		_onErrorRecovery = false;
		// consume buffer contents
		ConsumeBuffer(result.GetLastReadPosition(), end);
		// open new parse context
		try
		{
			StartBlockProcessing(_blockHeaderParser.GetBlockName(), _blockHeaderParser.GetParameterList());
		}
		catch (axis::foundation::AxisException&)
		{	// usually, a parser should not throw exceptions when loading context;
			// register event and suggest to abort operation
			RegisterErrorState(CriticalError);
			_parseContext.RegisterEvent(asmm::ErrorMessage(0x300507, 
        AXIS_ERROR_MSG_UNDEFINED_OPEN_PARSER_BEHAVIOR, asmm::ErrorMessage::ErrorCritical));
			return AbortRequest;
		}
		// we left block head reading process
		_onBlockHeadRead = false;
		return Accepted;
	}
	else if(result.GetResult() == aslp::ParseResult::FullReadPartialMatch)
	{	// found part of a block header
		_onBlockHeadRead = true;
		// do any pending error recovery steps
		TryRecoverySteps();
		// validate input buffer
		if (!ValidateInputBuffer())
		{	// validation failed, abort
			return AbortRequest;
		}
		// assume we are in the middle of a block head reading
		_onBlockHeadRead = true;
		return Accepted;
	}
	else if(result.GetResult() == aslp::ParseResult::FailedMatch && result.GetLastReadPosition() != begin)
	{	// that is, part of syntax was found, but something went wrong in the middle...
		// run error recovery steps
		RunRecoverySteps(result.GetLastReadPosition());
		_onBlockHeadRead = true;
		return SyntaxError;
	}

	// undoubtedly, it is not a header
	return NotAccepted;
}
aapc::StatementDecoder::ParsingState aapc::StatementDecoder::ForwardBufferParsingTask( 
  const asli::InputIterator& begin, const asli::InputIterator& end )
{
	try
	{
		aslp::ParseResult result = _currentContext->Parse(begin, end);
		if (result.IsMatch())
		{
			ConsumeBuffer(result.GetLastReadPosition(), end);
		}
		else if (result.GetResult() == aslp::ParseResult::FailedMatch)
		{
			RunRecoverySteps(result.GetLastReadPosition());
			return SyntaxError;
		}
	}
	catch (...)
	{
		// unexpected error occurred, abort operation
		RegisterErrorState(CriticalError);
		_parseContext.RegisterEvent(
      asmm::ErrorMessage(0x300508, AXIS_ERROR_MSG_UNEXPECTED_PARSER_READ_BEHAVIOR));

		// let's try not to repeat the same error; we will suppose that this error
		// was caused by something wrong in the input buffer, so we will clean it
		_parseBuffer.clear();
		_onErrorRecovery = false;

		return AbortRequest;
	}
	return Accepted;
}

aapc::StatementDecoder::ParsingState aapc::StatementDecoder::TestAndRunForBlockTail( 
  const asli::InputIterator& begin, const asli::InputIterator& end )
{
	aslp::ParseResult result = _blockTailParser(begin, end);
	if (result.IsMatch())
	{	// found a block tail
		// if an entire block got something wrong, here is the chance to flush the
		// entire buffer
		_onBlockTailRead = true;
		TryRecoverySteps();
		// clear error condition
		_onErrorRecovery = false;
		// consume buffer contents
		ConsumeBuffer(result.GetLastReadPosition(), end);
		// close parse context
		try
		{
			CloseBlockProcessing(_blockTailParser.GetBlockName());
		}
		catch (axis::foundation::AxisException&)
		{	// usually, a parser should not throw exceptions when unloading context;
			// register event and suggest to abort operation
			RegisterErrorState(CriticalError);
			_parseContext.RegisterEvent(asmm::ErrorMessage(0x300507, 
        AXIS_ERROR_MSG_UNDEFINED_OPEN_PARSER_BEHAVIOR));
			return AbortRequest;
		}
		// we left block head reading process
		_onBlockTailRead = false;
		return Accepted;
	}
	else if(result.GetResult() == aslp::ParseResult::FullReadPartialMatch)
	{	// found part of a block tail
		// do any pending error recovery steps
		TryRecoverySteps();

		// validate input buffer
		if (!ValidateInputBuffer())
		{	// validation failed, abort
			return AbortRequest;
		}

		// assume we are in the middle of a block tail reading
		_onBlockTailRead = true;

		return Accepted;
	}
	else if(result.GetResult() == aslp::ParseResult::FailedMatch 
          && result.GetLastReadPosition() != begin)
	{	// that is, part of syntax was found, but something went wrong in the middle...
		// run error recovery steps
		RunRecoverySteps(result.GetLastReadPosition());
		// we will assume that the user wanted to close the current block
		CloseBlockProcessing(_currentBlockName);
		_onBlockTailRead = false;
		return SyntaxError;
	}
	// undoubtedly, it is not a header
	return NotAccepted;
}

bool aapc::StatementDecoder::ProcessLine( void )
{
	if (_wasFinalized)
	{	// we can't process anything if we already have ended operation
		throw axis::foundation::InvalidOperationException();
	}
	ParsingState result;
	if (!ValidateInputBuffer())
	{	// buffer overflow condition detected
		_onErrorRecovery = false;
		return false;
	}

	/*
		We will try to read the buffer (or part of it) until no error condition
		is found (if no error is found at all, parsing succeeds at first strike).
		The error recovery method will always clear the error state at some
		point (even if it has to clear the entire buffer).
	*/
	do
	{
		// create iterators
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(_parseBuffer);
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(_parseBuffer.end());
		// PASS 1: let's check if it is a block header
		result = TestAndRunForBlockHeader(begin, end);
		if (result == Accepted)
		{
			_onErrorRecovery = false;
			return true;
		}
		else if(result == AbortRequest)
		{
			_onErrorRecovery = false;
			return false;
		}
		// PASS 2: let's check if it is a block tail
		if (result != SyntaxError)
		{
			result = TestAndRunForBlockTail(begin, end);
			if (result == Accepted)
			{
				_onErrorRecovery = false;
				return true;
			}
			else if(result == AbortRequest)
			{
				_onErrorRecovery = false;
				return false;
			}
		}
		// it is an instruction which must be processed by the current context
		// parser -- delegate this task to him
		if (result != SyntaxError)
		{
			result = ForwardBufferParsingTask(begin, end);
			if (result == Accepted)
			{
				_onErrorRecovery = false;
				return true;	// ok, a header was processed (somehow)
			}
			else if(result == AbortRequest)
			{
				_onErrorRecovery = false;
				return false;
			}
		}
	} while (_onErrorRecovery && !_parseBuffer.empty());
	return true;
}

void aapc::StatementDecoder::FeedDecoderBuffer( const axis::String& instructionChunk )
{
  // store new line in the buffer (add space to avoid unexpected tokens
  // agglutination)
  if (!instructionChunk.empty())
  {
    if(!_parseBuffer.empty()) _parseBuffer.append(' ');
    _parseBuffer.append(instructionChunk);
  }
}

void aapc::StatementDecoder::StartBlockProcessing( const axis::String& blockName, 
                                                   const aslse::ParameterList& paramList )
{
	aapps::BlockParser *nestedContext = NULL;
	aapps::BlockParser *previousParser = NULL;

	// try to get a nested parser; errors should be handled in the parser
	try
	{
		nestedContext = &_currentContext->GetNestedContext(blockName, paramList);
	}
	catch (axis::foundation::NotSupportedException&)
	{	// we couldn't find any parser suitable to this block
		RegisterErrorState(ParseError);
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x300509, AXIS_ERROR_MSG_UNKNOWN_BLOCK));
		_onBlockHeadRead = false;
		return;
	}
	catch (...)
	{	// couldn't open parser
		RegisterErrorState(ParseError);
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x30050A, AXIS_ERROR_MSG_GET_PARSER_FAILED));
		_onBlockHeadRead = false;
		return;
	}
	// push context to stack
	_contextStack.push_front(_currentContext);
	_openedContexts.insert(_currentContext);
	_blockNameHistory.push_front(_currentBlockName);
	_currentBlockName = blockName;
	previousParser = _currentContext;
	_currentContext = nestedContext;
	try
	{
		_currentContext->SetAnalysis(previousParser->GetAnalysis());
		_currentContext->StartContext(_parseContext);
	}
	catch (axis::foundation::AxisException& e)
	{
		throw axis::foundation::Level2ParserException(&e);
	}
	// tidy up everything
	_nestedBlockName.clear();
	_onBlockHeadRead = false;
}

void aapc::StatementDecoder::CloseBlockProcessing( axis::String& blockName )
{
	if (_contextStack.size() == 0)
	{	// we are at the top level, can't go any further
		RegisterErrorState(ParseError);
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x30050B, AXIS_ERROR_MSG_TAIL_IN_EXCESS));
		return;
	}

	if (_currentBlockName.compare(blockName) != 0)
	{	// whoops, wrong block name
		String s(AXIS_ERROR_MSG_BLOCK_DELIMITER_MISMATCH);
		s += blockName;
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x30050C, s));
		RegisterErrorState(ParseError);
	}

	try
	{	// even if it did not matched, close it anyway
		_currentContext->CloseContext();
	}
	catch (axis::foundation::AxisException&)
	{	// this shouldn't have happened...
		_parseContext.RegisterEvent(
      asmm::ErrorMessage(0x30050D, AXIS_ERROR_MSG_UNDEFINED_CLOSE_PARSER_BEHAVIOR));
		RegisterErrorState(CriticalError);
	}
	delete _currentContext;

	_currentContext = _contextStack.front();
	_contextStack.pop_front();

	_currentBlockName = _blockNameHistory.front();
	_blockNameHistory.pop_front();
}

void aapc::StatementDecoder::SetCurrentContext( aapps::BlockParser& parser )
{
	_currentContext = &parser;
}

void aapc::StatementDecoder::EndProcessing( void )
{
	if (_wasFinalized)
	{	// we already have ended processing; ignore it
		return;
	}

	if (_contextStack.size() > 0)
	{	// we have blocks that weren't closed correctly
		while (_contextStack.size() > 0)
		{
			// first, register the error
			String s(AXIS_ERROR_MSG_TAIL_MISSING);
			s += _currentBlockName;
			_parseContext.RegisterEvent(asmm::ErrorMessage(0x30050E, s));
			RegisterErrorState(ParseError);

			// then, close the block
			try
			{
				_currentContext->CloseContext();
			}
			catch (axis::foundation::AxisException&)
			{	// this shouldn't have happened...
				String s = AXIS_ERROR_MSG_UNDEFINED_CLOSE_PARSER_BEHAVIOR + _currentBlockName;
				_parseContext.RegisterEvent(asmm::ErrorMessage(0x30050D, s));
				RegisterErrorState(CriticalError);
			}
			delete _currentContext;

			// get next block
			_currentContext = _contextStack.front();
			_contextStack.pop_front();
			_currentBlockName = _blockNameHistory.front();
			_blockNameHistory.pop_front();
		}
	}

	// close the root block
	try
	{
		_currentContext->CloseContext();
	}
	catch (axis::foundation::AxisException&)
	{	// this shouldn't have happened...
		String s = AXIS_ERROR_MSG_UNDEFINED_CLOSE_PARSER_BEHAVIOR + String(_T("<root>"));
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x30050D, s));
		RegisterErrorState(CriticalError);
	}
	delete _currentContext;
	_currentBlockName = _T("");
	_wasFinalized = true;
}

bool aapc::StatementDecoder::ValidateInputBuffer( void )
{
	// avoid buffer inflating to more than 4 MB
	if (_parseBuffer.size() > _maxBufferLength)
	{
		RegisterErrorState(CriticalError);
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x30050F, AXIS_ERROR_MSG_STATEMENT_TOO_LONG, 
      asmm::ErrorMessage::ErrorCritical));
		_parseBuffer.clear();
		return false;
	}
	return true;
}

void aapc::StatementDecoder::ConsumeBuffer( const asli::InputIterator& lastReadPos, 
                                            const asli::InputIterator& bufferEndPos )
{
	_parseBuffer = lastReadPos.ToString(bufferEndPos);
}

void aapc::StatementDecoder::RunRecoverySteps( const asli::InputIterator& lastReadPosition )
{
	String symbol;
	asli::InputIterator end = aslf::IteratorFactory::CreateStringIterator(_parseBuffer.end());

	if (lastReadPosition == end)
	{	// there is no more symbol to throw away; flush buffer
		_parseBuffer.clear();
		_onErrorRecovery = false;
		return;
	}
	_onErrorRecovery = true;
	// extract misplaced token
	asli::InputIterator symbolEnd = aslf::AxisGrammar::ExtractNextToken(lastReadPosition, end);
	symbol = lastReadPosition.ToString(symbolEnd);

	// log event
	if (aslf::AxisGrammar::IsValidToken(symbol))
	{
		String s(AXIS_ERROR_MSG_MISPLACED_SYMBOL);
		s += symbol;
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x300510, s));
		RegisterErrorState(ParseError);
	}
	else
	{
		String s(AXIS_ERROR_MSG_INVALID_CHAR_SEQUENCE);
		s += symbol;
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x300511, s));
		RegisterErrorState(ParseError);
	}

	// throw away symbol from the buffer
	ConsumeBuffer(symbolEnd, end);
}

void aapc::StatementDecoder::TryRecoverySteps( void )
{
	if (_onBlockHeadRead && _onBlockTailRead && _onErrorRecovery)
	{	// we couldn't read head but we found a tail; ignore the entire head
		_onBlockHeadRead = false;
		_parseContext.RegisterEvent(asmm::ErrorMessage(0x300512, AXIS_ERROR_MSG_UNPROCESSED_BLOCK));
		RegisterErrorState(ParseError);
		_onErrorRecovery = false;
	}
}

bool aapc::StatementDecoder::IsBufferEmpty( void ) const
{
	return _parseBuffer.empty();
}

aapc::StatementDecoder::ErrorState aapc::StatementDecoder::GetErrorState( void ) const
{
	return _errorState;
}

axis::String aapc::StatementDecoder::GetBufferContents( void ) const
{
	return _parseBuffer;
}

void aapc::StatementDecoder::RegisterErrorState( ErrorState state )
{
	if ((int)state > (int)_errorState)
	{
		_errorState = state;
	}
}

unsigned long aapc::StatementDecoder::GetMaxBufferLength( void ) const
{
	return _maxBufferLength;
}

void aapc::StatementDecoder::SetMaxBufferLength( unsigned long length )
{
	if (length == 0)
	{
		throw axis::foundation::ArgumentException();
	}
	_maxBufferLength = length;
}

unsigned long aapc::StatementDecoder::GetBufferSize( void ) const
{
	return (unsigned long)_parseBuffer.size();
}

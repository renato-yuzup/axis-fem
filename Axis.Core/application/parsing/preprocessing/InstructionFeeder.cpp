#include "InstructionFeeder.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "foundation/OpenCommentException.hpp"
#include "foundation/IOException.hpp"
#include "foundation/EOFException.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "application/parsing/error_messages.hpp"

namespace aapc = axis::application::parsing::core;
namespace aapp = axis::application::parsing::preprocessing;
namespace asio = axis::services::io;
namespace asmm = axis::services::messaging;
namespace afd = axis::foundation::definitions;

aapp::InstructionFeeder::InstructionFeeder(aapc::ParseContext& context) :
_parseContext(context)
{
	_source = NULL;
	_eofFlag = false;
	_lastLineRead.clear();
	_isReadingComment = false;
	_lastLineNumber = 0;
}

aapp::InstructionFeeder::InstructionFeeder(asio::StreamReader& source, aapc::ParseContext& context) :
_parseContext(context)
{
	_source = &source;
	_eofFlag = false;
	_lastLineRead.clear();
	_isReadingComment = false;
	_lastLineNumber = 0;
	// look ahead for the next line
	ReadNextRelevantLine();
}

aapp::InstructionFeeder::~InstructionFeeder(void)
{
	// no-op
}

bool aapp::InstructionFeeder::IsEOF( void ) const
{
	return _eofFlag;
}

void aapp::InstructionFeeder::ReadLine( axis::String& line )
{
	if (IsEOF())
	{
		throw axis::foundation::EOFException();
	}
	line = _lastLineRead;
	_lastLineNumber = _source->GetLastLineNumber();
	ReadNextRelevantLine();
}

void aapp::InstructionFeeder::ReadNextRelevantLine( void )
{
	if (_source->IsEOF())
	{
		// there is no more line to read
		_eofFlag = true;
		return;
	}
	// read line
	String line;
	bool _readingCommentState = _isReadingComment;

	do
	{
		if (_isReadingComment != _readingCommentState && _isReadingComment)
		{	// we found a line where a comment block is starting
			_blockStartingLine = _source->GetLastLineNumber();
		}
		_readingCommentState = _isReadingComment;

		if (_source->IsEOF())
		{
			if(_isReadingComment)
			{	// we can't attempt to read any further, there is an open comment block...
				// throw an exception
				_parseContext.RegisterEvent(asmm::ErrorMessage(0x30061E, AXIS_ERROR_MSG_UNCLOSED_COMMENT_BLOCK));
			}
			_eofFlag = true;
			return;
		}
		try
		{
			_source->ReadLine(line);
		}
		catch (axis::foundation::IOException&)
		{	// problems reading from file; notify and abort
			_parseContext.RegisterEvent(
        asmm::ErrorMessage(0x30061F, AXIS_ERROR_MSG_INCLUDE_FILE_IO_ERROR + _source->GetStreamPath()));
			_eofFlag = true;
			break;
		}
	} while (!ParseLine(line));
}

bool aapp::InstructionFeeder::ParseLine( axis::String& line )
{
	axis::String extractedExpression;
	axis::String formattedLine = line;
	axis::String::iterator it = line.begin();
	bool result;
	axis::String::size_type position, aux;

	// check if it is a blank line
	StringServices::Trim(extractedExpression, line);
	result = (extractedExpression.size() == 0);
	extractedExpression.clear();	// tidy our temporary work

	if (result) return false; // ask for a valid line

	position = line.find(afd::AxisInputLanguage::EndBlockComment, 0);
	if (_isReadingComment)
	{
		if(position == line.npos)
		{
			return false; // still inside the comment
		}
		else
		{
			// exit block comment mode
			_isReadingComment = false;

			// parse the remaining characters
			return ParseLine(line.substr(position + String(afd::AxisInputLanguage::EndBlockComment).size()));
		}
	}
	else
	{
		if (position != line.npos)
		{
			// it wasn't previously indicated that a comment block was opened;
			// however, the code can still be syntactically correct in two
			// situations:
			bool cond1, cond2;

			// 1) the end comment appear after a begin comment, in the same line
			aux = line.find(afd::AxisInputLanguage::BeginBlockComment, 0);
			cond1 = (aux != line.npos && aux < position);

			// 2) the end comment indicator is presented inside an inline comment
			aux = line.find(afd::AxisInputLanguage::InlineComment, 0);
			cond2 = (position != line.npos &&  aux < position);

			// if none of these conditions were satisfied, then it is a syntax
			// error
			if (!(cond1 || cond2))
			{
				_parseContext.RegisterEvent(asmm::ErrorMessage(0x30061D, AXIS_ERROR_MSG_END_COMMENT_IN_EXCESS));
				return ParseLine(line.substr(position + String(afd::AxisInputLanguage::EndBlockComment).size()));
			}
		}
	}

	// remove inline block comments
	while((position = formattedLine.find(afd::AxisInputLanguage::BeginBlockComment, 0)) != 
         formattedLine.npos)
	{
		aux = formattedLine.find(afd::AxisInputLanguage::InlineComment, 0);
		if (aux < position)
		{	// we reached a fake condition; the start delimiter is inside an inline
			// comment; anything after this is just a false positive, so we ignore it
			break;
		}

		extractedExpression.append(formattedLine.substr(0, position));
		formattedLine = 
      formattedLine.substr(position + String(afd::AxisInputLanguage::BeginBlockComment).size());
		position = formattedLine.find(afd::AxisInputLanguage::EndBlockComment, 0);
		if (position != line.npos)
		{
			formattedLine = formattedLine.substr(
        position + String(afd::AxisInputLanguage::EndBlockComment).size(), formattedLine.npos);
			position = formattedLine.find(afd::AxisInputLanguage::BeginBlockComment, 0);
			if (extractedExpression.length() > 0)
			{
				if (extractedExpression[extractedExpression.length() - 1] != ' ' && 
            extractedExpression[extractedExpression.length() - 1] != '\t')
				{
					extractedExpression += _T(' ');	 // just to ensure separation of terms
				}
			}
		}
		else
		{	// we found the start of a block comment, remember to ignore comment lines
			_isReadingComment = true;
		}
	}

	// add the remaining of the expression, if there is any
	if (!_isReadingComment)
	{
		extractedExpression.append(formattedLine);
	}

	// remove inline comment, if any
	position = extractedExpression.find(afd::AxisInputLanguage::InlineComment, 0);
	if(position != extractedExpression.npos)
	{
		extractedExpression= extractedExpression.substr(0, position);
	}
	// store expression read without white spaces
	StringServices::Trim(_lastLineRead, extractedExpression);
	return (_lastLineRead.size() > 0); // success only if it not resulted in a blank line
}

asio::StreamReader& aapp::InstructionFeeder::GetCurrentSource( void ) const
{
	return *_source;
}

void aapp::InstructionFeeder::ChangeSource( asio::StreamReader& reader )
{
	if (_source == &reader)
	{	// ignore it
		return;
	}
	// change target
	_source = &reader;
	_eofFlag = false;
	// look ahead for the next line
	ReadNextRelevantLine();
}

void aapp::InstructionFeeder::Rewind( void )
{
	// rewind stream state, if exists
	if (_source != NULL)
	{
		_source->PushBackLine();
	}
}

unsigned long aapp::InstructionFeeder::GetLastLineReadIndex( void ) const
{
	return _lastLineNumber;
}

void aapp::InstructionFeeder::Reset( void )
{
	_source = NULL;
	_eofFlag = false;
	_lastLineRead.clear();
	_isReadingComment = false;
	_lastLineNumber = 0;
}

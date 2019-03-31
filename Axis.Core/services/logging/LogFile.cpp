#include "LogFile.hpp"
#include "services/io/FileWriter.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "services/messaging/metadata/SourceFileMetadata.hpp"
#include "foundation/Assertion/assertion.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace asio  = axis::services::io;
namespace asl   = axis::services::locales;
namespace aslg  = axis::services::logging;
namespace asmm  = axis::services::messaging;
namespace asmmm = axis::services::messaging::metadata;
namespace af    = axis::foundation;

namespace {
const axis::String::char_type * ErrorString = _T("<ERROR> ");
const axis::String::char_type * InfoString  = _T("<INFO > ");
const axis::String::char_type * WarnString  = _T("<WARN > ");
const axis::String::char_type * LogDefaultPadding = _T("   ");
const int ExceptionIndent = 20;
const int InnerExceptionIndent = 2;
}

aslg::LogFile::LogFile( const axis::String& filename, bool appendMode ) :
_currentLocale(asl::LocaleLocator::GetLocator().GetGlobalLocale())
{
	_writer = OpenFileStream(filename, appendMode);
	_writeMode = appendMode? asio::StreamWriter::kAppend : asio::StreamWriter::kOverwrite;
	_isInsideSection = false;
	_nestingIndex = 0;
	_onRootSection = false;
	_padding = LogDefaultPadding;
}

aslg::LogFile::LogFile( const axis::String& filename ) :
_currentLocale(asl::LocaleLocator::GetLocator().GetGlobalLocale())
{
	_writer = OpenFileStream(filename);
	_writeMode = asio::StreamWriter::kAppend;
	_isInsideSection = false;
	_nestingIndex = 0;
	_onRootSection = false;
	_padding = LogDefaultPadding;
}

aslg::LogFile::LogFile( asio::StreamWriter& writer ) :
_currentLocale(asl::LocaleLocator::GetLocator().GetGlobalLocale())
{
	_writer = &writer;
	_writeMode = asio::StreamWriter::kAppend;
	_isInsideSection = false;
	_nestingIndex = 0;
	_onRootSection = false;
	_padding = LogDefaultPadding;
}

aslg::LogFile::~LogFile( void )
{
	_writer->Destroy();
	_writer = nullptr;
}

aslg::LogFile& aslg::LogFile::Create( const axis::String& filename, bool appendMode )
{
	return *new LogFile(filename, appendMode);
}

aslg::LogFile& aslg::LogFile::Create( const axis::String& filename )
{
	return *new LogFile(filename);
}

aslg::LogFile& aslg::LogFile::Create( asio::StreamWriter& writer )
{
	return *new LogFile(writer);
}

void aslg::LogFile::Destroy( void ) const
{
	delete this;
}

asio::StreamWriter * aslg::LogFile::OpenFileStream( const axis::String& filename, 
                                                    bool appendMode /*= true*/ ) const
{
	asio::FileWriter *writer = &asio::FileWriter::Create(filename);
	return writer;
}

void aslg::LogFile::StartLogging( void )
{
	if (_writer->IsOpen())
	{
		throw axis::foundation::InvalidOperationException();
	}
	_writer->Open(_writeMode, asio::StreamWriter::kSharedMode);
}

bool aslg::LogFile::IsReady( void ) const
{
	return _writer->IsOpen();
}

void aslg::LogFile::PrintHeader( const axis::String& heading ) const
{
	PrintHeader(heading, '#');
}

void aslg::LogFile::PrintHeader( const axis::String& heading, 
                                 const axis::String::char_type pattern ) const
{
	int headingLength = (int)heading.size() > 128? (int)heading.size() + 6 : 128;
	int numSpaces = (headingLength - (int)heading.size() - 4) / 2;
	String separatorLine(headingLength, pattern);
	String spaces(numSpaces, ' ');
	String headLine = String(2, pattern) + spaces + heading + spaces;
	if (headLine.size() + 2 < headingLength)
	{
		headLine += String(' ');
	}
	headLine += String(2, pattern);
	// write header to file
	_writer->WriteLine(separatorLine);
	_writer->WriteLine(headLine);
	_writer->WriteLine(separatorLine);
}

void aslg::LogFile::StopLogging( void )
{
	if (!_writer->IsOpen())
	{
		throw axis::foundation::InvalidOperationException();
	}
	_writer->Close();
}

void aslg::LogFile::PrintException( const axis::foundation::AxisException& e ) const
{
	String spaces(ExceptionIndent, ' ');
	const axis::foundation::AxisException *ex = &e;
	String line = spaces + _T("Triggered exception -- ");
	while (ex != NULL)
	{
		line += ex->GetTypeName() + _T(":");
		if (!ex->GetMessage().empty())
		{
			line += _T(" ") + ex->GetMessage();
		}
		_writer->WriteLine(line);

		ex = e.GetInnerException();
		if (ex!= NULL)
		{
			spaces += String(InnerExceptionIndent, ' ');
			line = spaces + _T("Inner Exception ");
		}
	}
}

void aslg::LogFile::DoProcessEventMessage( asmm::EventMessage& volatileMessage )
{
	if (!_writer->IsOpen())
	{
		throw axis::foundation::InvalidOperationException();
	}
	if (volatileMessage.IsInfo())
	{
		LogInfoMessage((asmm::InfoMessage&) volatileMessage);
	}
	else if (volatileMessage.IsWarning())
	{
		LogWarningMessage((asmm::WarningMessage&)volatileMessage);
	}
	else if (volatileMessage.IsError())
	{
		LogErrorMessage((asmm::ErrorMessage&)volatileMessage);
	}
	else if (volatileMessage.IsLogEntry())
	{
		WriteLogMessage((asmm::LogMessage&)volatileMessage);
	}
	_writer->Flush();
}

void aslg::LogFile::LogInfoMessage( asmm::InfoMessage& message ) const
{
  auto& dateTimeLocale = _currentLocale.GetDataTimeLocale();
	String timestamp = 
    _T("[") + dateTimeLocale.ToShortDateTimeMillisString(message.GetTimestamp()) + _T("]");
	String s = _padding + timestamp + _T("\t") + InfoString;
	if (message.GetId() != 0)
	{
		s += _T("(0x") + String::int_to_hex((long)message.GetId()) + _T(") ");
	}
	s += message.GetDescription();
	_writer->WriteLine(s);
}

void aslg::LogFile::LogWarningMessage( axis::services::messaging::WarningMessage& message ) const
{
  auto& dateTimeLocale = _currentLocale.GetDataTimeLocale();
	String timestamp = 
    _T("[") + dateTimeLocale.ToShortDateTimeMillisString(message.GetTimestamp()) + _T("]");
	String s = _padding + timestamp + _T("\t") + WarnString;
	if (message.GetId() != 0)
	{
		s += _T("(0x") + String::int_to_hex((long)message.GetId()) + _T(") ");
	}
	s += message.GetDescription();
	_writer->WriteLine(s);
	if (message.HasAssociatedException())
	{
		PrintException(message.GetAssociatedException());
	}
}

void aslg::LogFile::LogErrorMessage( asmm::ErrorMessage& message ) const
{
  auto& dateTimeLocale = _currentLocale.GetDataTimeLocale();
	String timestamp = _T("[") + dateTimeLocale.ToShortDateTimeMillisString(message.GetTimestamp()) + _T("]");
	String s = _padding + timestamp + _T("\t") + ErrorString;
	if (message.GetId() != 0)
	{
		s += _T("(0x") + String::int_to_hex((long)message.GetId()) + _T(") ");
	}
	String sourceInfo = BuildSourceInformationString(message);
	s += message.GetDescription() + sourceInfo;
	_writer->WriteLine(s);
	if (message.HasAssociatedException())
	{
		PrintException(message.GetAssociatedException());
	}
}

void aslg::LogFile::WriteLogMessage( asmm::LogMessage& message )
{
	if (message.IsLogCommand())
	{	
		if (message.DoesStartNewSection())
		{	// jump two lines
			if (!_isInsideSection)
			{
				_onRootSection = false;
				_isInsideSection = true;
				_nestingIndex = 0;

				_writer->WriteLine(_T("  ") + message.GetDescription());
				_writer->WriteLine(); 
			}
		}
		else if (message.DoesCloseSection())
		{
			if (_isInsideSection)
			{
				_isInsideSection = false;
				_nestingIndex = 0;

				_writer->WriteLine(_T("  ") + message.GetDescription());
				_writer->WriteLine(); 
			}
		}
		else if (message.DoesStartNewBlock())
		{	// jump one line
			if (!_onRootSection) _writer->WriteLine();
		}
		else if (message.DoesCloseBlock())
		{	// jump one line
			if (!_onRootSection) _writer->WriteLine();
		}
		else if (message.DoesStartNewNesting())
		{
			if (!_onRootSection) _nestingIndex++;
		}
		else if (message.DoesCloseNesting())
		{
			if (!_onRootSection) if (_nestingIndex > 0) _nestingIndex--;
		}
		else if (message.DoesStartNewBanner())
		{
			_onRootSection = true;
			_isInsideSection = false;
			_nestingIndex = 0;
		}
		else if (message.DoesCloseBanner())
		{
			_onRootSection = false;
		}
		// calculate new padding
		if (_onRootSection)
		{
			_padding.clear();
		}
		else
		{
			int paddingSize = 2 + 3 * _nestingIndex;
			if (_isInsideSection) paddingSize += 3;
			_padding = String(paddingSize, ' ');
		}
	}
	else
	{
		_writer->WriteLine(_padding + message.GetDescription());
	}
}

void aslg::LogFile::PrintLine( const axis::String& line )
{
	_writer->WriteLine(line);
}

axis::String aslg::LogFile::BuildSourceInformationString( const asmm::Message& message ) const
{
	// check if message has source information
	if (message.GetMetadata().Contains(asmmm::SourceFileMetadata::GetClassName()))
	{	
		asmmm::SourceFileMetadata& sfm = safe_cast<asmmm::SourceFileMetadata>(message.GetMetadata()[
      asmmm::SourceFileMetadata::GetClassName()]);
		return _T(" (") + sfm.GetSourceFileLocation() + _T(" at line ") + 
           String::int_parse(sfm.GetLineIndex());
	}
	return _T("");
}

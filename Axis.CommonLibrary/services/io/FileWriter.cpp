#include "FileWriter.hpp"
#include <stdexcept>

axis::services::io::FileWriter::FileWriter( const axis::String& fileName ) :
_fileName(fileName)
{
	_stream = new FileStreamWriter(fileName);
	Init();
}

axis::services::io::FileWriter::~FileWriter( void )
{
	_stream->Destroy();
  _stream = NULL;
}

void axis::services::io::FileWriter::WriteLine( const axis::String& s )
{
	_stream->Write(s + _eol);
}

void axis::services::io::FileWriter::WriteLine( void )
{
	_stream->Write(_eol);
}

void axis::services::io::FileWriter::Write( const axis::String& s )
{
	_stream->Write(s);
}

axis::String axis::services::io::FileWriter::GetEndOfLineSequence( void ) const
{
	return _eol;
}

void axis::services::io::FileWriter::SetEndOfLineSequence( const axis::String& eol )
{
	_eol = eol;
}

unsigned long axis::services::io::FileWriter::GetBytesWritten( void ) const
{
	return _bytesWritten;
}

bool axis::services::io::FileWriter::IsAutoFlush( void ) const
{
	// TODO: modify this in the future
	return false;
}

bool axis::services::io::FileWriter::IsBuffered( void ) const
{
	// TODO: modify this in the future
	return false;
}

unsigned long axis::services::io::FileWriter::GetBufferSize( void ) const
{
	// TODO: modify this in the future
	throw std::exception("The method or operation is not implemented.");
}

unsigned long axis::services::io::FileWriter::GetBufferUsedSpace( void ) const
{
	// TODO: modify this in the future
	throw std::exception("The method or operation is not implemented.");
}

bool axis::services::io::FileWriter::IsOpen( void ) const
{
	return _isOpened;
}

void axis::services::io::FileWriter::Flush( void )
{
	_stream->Flush();
}

void axis::services::io::FileWriter::ToggleFlush( void )
{
	// TODO: modify this in the future
	// nothing to do here
}

void axis::services::io::FileWriter::Close( void )
{
	_stream->Close();
	_isOpened = false;
}

axis::String axis::services::io::FileWriter::GetStreamPath( void ) const
{
	return _fileName;
}

void axis::services::io::FileWriter::Open( WriteMode writeMode /*= WriteMode::Overwrite*/, LockMode lockMode /*= LockMode::SharedMode */ )
{
	FileStreamWriter::WriteMode os_writeMode;
	FileStreamWriter::LockMode os_lockMode;

	switch (writeMode)
	{
	case kAppend:
		os_writeMode = FileStreamWriter::Append;
		break;;
	default:	// Overwrite
		os_writeMode = FileStreamWriter::Overwrite;
	}

	switch (lockMode)
	{
	case kExclusiveMode:
		os_lockMode = FileStreamWriter::ExclusiveMode;
		break;
	default: // SharedMode
		os_lockMode = FileStreamWriter::SharedMode;
	}

	_stream->Open(os_writeMode, os_lockMode);
	_isOpened = true;
}

void axis::services::io::FileWriter::Destroy( void ) const
{
	if (_isOpened)
	{
		_stream->Close();
		_stream->Destroy();
	}
}

void axis::services::io::FileWriter::Init( void )
{
	_eol = _T("\n");
	_bytesWritten = 0;
	_isOpened = false;
}

axis::services::io::FileWriter& axis::services::io::FileWriter::Create( const axis::String& fileName )
{
	return *new axis::services::io::FileWriter(fileName);
}
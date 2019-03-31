#include "FileReader.hpp"
#include <string>
#include "foundation/NotSupportedException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "foundation/IOException.hpp"
#include "string_traits.hpp"
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#define AXIS_INPUTFILE_BUFFER_LENGTH		4096*1024	// 4 MB

namespace asio = axis::services::io;

static const int lineStringBufferReserve = 4096;

class asio::FileReader::InternalFileData
{
public:
  typedef axis::String::char_type string_unit;
  typedef std::basic_ifstream<axis::String::char_type> input_stream;
  typedef std::basic_filebuf<axis::String::char_type> input_stream_buffer;

  typedef boost::iostreams::stream_buffer<boost::iostreams::mapped_file_source> file_buffer;

  InternalFileData(const axis::String& filename);
  void Init(const axis::String& filename);

  axis::String fileName;
  input_stream fileStream;
//   file_buffer fileStream;
  string_unit * buffer;
  unsigned long lastLineReadIdx;
  bool wasLineRead;
  axis::String lastLineRead;
  bool hasPushedBackLine;
};

asio::FileReader::InternalFileData::InternalFileData( const axis::String& filename ) :
  fileName(filename), buffer(NULL)
{
  lastLineRead.reserve(lineStringBufferReserve);
  Init(fileName);
}

void asio::FileReader::InternalFileData::Init( const axis::String& filename )
{
  // try to allocate read buffer
  try
  {
    buffer = new string_unit[AXIS_INPUTFILE_BUFFER_LENGTH];		
  }
  catch(...)
  {
    throw axis::foundation::OutOfMemoryException();
  }

  // select read buffer
  input_stream_buffer *buf = fileStream.rdbuf();
  buf->pubsetbuf(buffer, AXIS_INPUTFILE_BUFFER_LENGTH);

  // initialize variables
  lastLineReadIdx = 0;
  wasLineRead = false;
  hasPushedBackLine = false;

  // try to open the file
  if(buf->open(filename.data(), std::ios_base::in) == NULL)
  {
    // an error occurred while trying to open; abort
    buf->pubsetbuf(NULL, 0);
    delete[] buffer;

    throw axis::foundation::IOException();
  }
}

asio::FileReader::FileReader(const axis::String& filename)
{
	data_ = new InternalFileData(filename);
}

asio::FileReader::~FileReader(void)
{
	Close();
}

void asio::FileReader::ReadLine(axis::String& s)
{
	if (IsEOF())
	{	// passed the end of the file; we can't read from here
		throw axis::foundation::IOException();
	}
	if (data_ == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}

	if(data_->hasPushedBackLine)
	{
		data_->wasLineRead = true;
		s = data_->lastLineRead;
		data_->hasPushedBackLine = false;
		(data_->lastLineReadIdx)++;
	}
	else
	{
		try
		{
			// try to read a line from the file
			axis::getline(data_->fileStream, data_->lastLineRead);
			s = data_->lastLineRead;
			(data_->lastLineReadIdx)++;
			data_->wasLineRead = true;
		}
		catch(...)
		{
			throw axis::foundation::IOException();
		}
	}
}

void asio::FileReader::PushBackLine( void )
{
	if (data_->hasPushedBackLine) throw axis::foundation::NotSupportedException();
	if(!data_->wasLineRead) throw axis::foundation::InvalidOperationException();

	// ignore if it's EOF
	if (IsEOF()) return;

	data_->hasPushedBackLine = true;
	(data_->lastLineReadIdx)--;
}

bool asio::FileReader::IsEOF(void) const
{
	if (data_ == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return IsOpen()? data_->fileStream.eof() : false;
}

bool asio::FileReader::IsOpen(void) const
{
	return (data_ != NULL) && (data_->fileStream.is_open());
}

void asio::FileReader::Close(void)
{	
	if(IsOpen())
	{	// clean resources
		try
		{
			data_->fileStream.close();
		}
		catch (...)
		{
			throw axis::foundation::IOException();
		}
		data_->fileStream.rdbuf()->pubsetbuf(NULL, 0);
		delete[] data_->buffer;
		delete data_;
    data_ = NULL;
	}
}

axis::String asio::FileReader::GetStreamPath(void) const
{
	return data_->fileName;
}

unsigned long asio::FileReader::GetLastLineNumber( void ) const
{
	return data_->lastLineReadIdx;
}

void asio::FileReader::Reset( void )
{
	axis::String filename = data_->fileName;
	Close();
	data_ = new InternalFileData(filename);
}


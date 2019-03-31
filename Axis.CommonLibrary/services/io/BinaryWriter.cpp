#include "BinaryWriter.hpp"
#include <stdio.h>
#include "foundation/InvalidOperationException.hpp"
#include "foundation/IOException.hpp"

// #define SWAP_ENDIAN(c) ((c) & 0x00FF) * 0x100 + ((c) & 0xFF00) / 0x100
#define SWAP_ENDIAN(c) c

namespace asio = axis::services::io;

static const int defaultBufferLength = 40*1024*1024;

namespace {
  inline int get_encoding_length(unsigned int c)
  {
    return (c < 0x80)? 1 : (c < 0x0800)? 2 : (c < 0x010000)? 3 : 4;
  }

  inline void utf8encode(char *output, unsigned int codepoint)
  {
    if (codepoint < 0x80)
    {
      output[0] = (char)codepoint;
    }
    else if (codepoint < 0x0800)
    {
      output[0] = codepoint >> 6  & 0x1F | 0xC0;
      output[1] = codepoint >> 0  & 0x3F | 0x80;
    }
    else if (codepoint < 0x010000)
    {
      output[0] = codepoint >> 12 & 0x0F | 0xE0;
      output[1] = codepoint >> 6  & 0x3F | 0x80;
      output[2] = codepoint >> 0  & 0x3F | 0x80;
    }
    else if (codepoint < 0x110000)
    {
      output[0] = codepoint >> 18 & 0x07 | 0xF0;
      output[1] = codepoint >> 12 & 0x3F | 0x80;
      output[2] = codepoint >> 6  & 0x3F | 0x80;
      output[3] = codepoint >> 0  & 0x3F | 0x80;
    }
  }
}

asio::BinaryWriter::BinaryWriter( const axis::String& fileName ) :
  fileName_(fileName), file_(nullptr), isOpened_(false), bytesWritten_(0), 
  bufferCursor_(0)
{
  Init(fileName, defaultBufferLength);
}

asio::BinaryWriter::BinaryWriter( const axis::String& fileName, int bufferLength ) :
  fileName_(fileName), file_(nullptr), isOpened_(false), bytesWritten_(0),
  bufferCursor_(0)
{
  Init(fileName, bufferLength);
}

void asio::BinaryWriter::Init( const axis::String& fileName, int bufferLength )
{
  eol_ = _T("\n");
  buffer_ = new char[bufferLength];  
  bufferSize_ = bufferLength;
  bufferCursor_ = 0;
}

asio::BinaryWriter::~BinaryWriter(void)
{
  if (isOpened_) Close();
  delete [] buffer_;
}

void asio::BinaryWriter::Destroy( void ) const
{
  delete this;
}

asio::BinaryWriter& asio::BinaryWriter::Create( const axis::String& fileName )
{
  return *new asio::BinaryWriter(fileName);
}

asio::BinaryWriter& asio::BinaryWriter::Create( const axis::String& fileName, int bufferLength )
{
  return *new asio::BinaryWriter(fileName, bufferLength);
}

bool asio::BinaryWriter::IsOpen( void ) const
{
  return isOpened_;
}

void asio::BinaryWriter::Open( WriteMode writeMode /*= kOverwrite*/, LockMode lockMode /*= kSharedMode */ )
{
  if (isOpened_)
  {
    throw axis::foundation::InvalidOperationException(_T("File is already opened."));
  }
  // determine file mode and open it
  axis::String mode;
  switch (writeMode)
  {
  case axis::services::io::StreamWriter::kOverwrite:
    mode = _T("wb");
    break;
  case axis::services::io::StreamWriter::kAppend:
    mode = _T("ab");
    break;
  default:
    // nothing to do here
    break;
  }
  FILE *file = _wfopen(fileName_.c_str(), mode.c_str());
  if (file == nullptr)
  {
    throw axis::foundation::IOException(_T("Couldn't open file for binary write operation."));
  }
  file_ = file;
  isOpened_ = true;
}

void asio::BinaryWriter::Close( void )
{
  if (!isOpened_)
  {
    throw axis::foundation::InvalidOperationException(_T("File has not been opened."));
  }
  if (bufferCursor_ > 0) Flush();
  fclose((FILE *)file_);
  isOpened_ = false;
}

unsigned long asio::BinaryWriter::GetBytesWritten( void ) const
{
  return bytesWritten_;
}

bool asio::BinaryWriter::IsAutoFlush( void ) const
{
  return true;
}

bool asio::BinaryWriter::IsBuffered( void ) const
{
  return true;
}

unsigned long asio::BinaryWriter::GetBufferSize( void ) const
{
  return bufferSize_;
}

unsigned long asio::BinaryWriter::GetBufferUsedSpace( void ) const
{
  return bufferCursor_;
}

axis::String asio::BinaryWriter::GetEndOfLineSequence( void ) const
{
  return eol_;
}

void asio::BinaryWriter::SetEndOfLineSequence( const axis::String& eol )
{
  eol_ = eol;
}

axis::String asio::BinaryWriter::GetStreamPath( void ) const
{
  return fileName_;
}

void asio::BinaryWriter::ToggleFlush( void )
{
  // this method is not supported; the operation does nothing
}

void asio::BinaryWriter::WriteLine( const axis::String& s )
{
  Write(s);
  Write(eol_);
}

void asio::BinaryWriter::WriteLine( void )
{
  Write(eol_);
}

void asio::BinaryWriter::Write( const axis::String& s )
{
  const axis::String::char_type *str = s.c_str();
  uint64 len = s.length();
  uint64 charsWritten = 0;
  while (charsWritten < len)
  {
    unsigned int c = str[charsWritten];
    int extractedBytes = 1;
#ifdef UNICODE
    // invert byte order
    c = SWAP_ENDIAN(c);
    if (c >= 0xD800 && c < 0xE000)
    { // codepoint located in the supplementary plane, get trail surrogate
      unsigned int trail = s[charsWritten + 1];
      trail = SWAP_ENDIAN(trail);
      trail -= 0xDC00;
      unsigned int lead = c - 0xD800;

      // calculate correct codepoint
      c = lead * 0x400 + trail;
      extractedBytes++;
    }
#endif
    int charsToWrite = get_encoding_length(c);
    if (bufferCursor_ + charsToWrite > bufferSize_)
    { // buffer will overflow; flush contents before continue
      Flush();
    }
    utf8encode(&buffer_[bufferCursor_], c);
    bufferCursor_ += charsToWrite;
    charsWritten += extractedBytes;
    bytesWritten_ += charsToWrite;
  }
}

void asio::BinaryWriter::Flush( void )
{
  if (bufferCursor_ > 0)
  {
    fwrite(buffer_, sizeof(char), bufferCursor_, (FILE *)file_);
    bufferCursor_ = 0;
  }
}

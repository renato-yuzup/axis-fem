#include "BinaryReader.hpp"
#include <stdio.h>
#include "foundation/IOException.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace asio = axis::services::io;

static const int defaultBufferSize = 40*1024*1024;
static const axis::String::char_type eol = '\n';

asio::BinaryReader::BinaryReader( const axis::String& filename ) :
  filename_(filename)
{
  Init(defaultBufferSize);
}

asio::BinaryReader::BinaryReader( const axis::String& filename, int bufferLength ) :
  filename_(filename)
{
  Init(bufferLength);
}

void asio::BinaryReader::Init( int bufferLength )
{
  buffer_ = new char[bufferLength];
  bufferSize_ = bufferLength;
  bufferCursor_ = bufferSize_;
  bufferContentLen_ = 0;
  curLineNumber_ = 0;
  isOpened_ = false;

  FILE *file = _wfopen(filename_, _T("rb"));
  if (file == nullptr)
  {
    throw axis::foundation::IOException(_T("Couldn't open file for read operation."));
  }
  file_ = file;
  isOpened_ = true;
  FillBuffer();
}

asio::BinaryReader& asio::BinaryReader::Create( const axis::String& filename )
{
  return *new asio::BinaryReader(filename);
}

asio::BinaryReader& asio::BinaryReader::Create( const axis::String& filename, int bufferLength )
{
  return *new asio::BinaryReader(filename, bufferLength);
}

asio::BinaryReader::~BinaryReader( void )
{
  if (isOpened_) Close();
  delete [] buffer_;
}

void asio::BinaryReader::Destroy( void )
{
  delete this;
}

bool asio::BinaryReader::IsOpen( void ) const
{
  return isOpened_;
}

axis::String asio::BinaryReader::GetStreamPath( void ) const
{
  return filename_;
}

unsigned long asio::BinaryReader::GetLastLineNumber( void ) const
{
  return curLineNumber_;
}

void asio::BinaryReader::Close( void )
{
  if (!isOpened_)
  {
    throw axis::foundation::InvalidOperationException(_T("File is not opened."));
  }
  fclose((FILE *)file_);
  isOpened_ = false;
}

bool asio::BinaryReader::IsEOF( void ) const
{
  if (!isOpened_)
  {
    throw axis::foundation::InvalidOperationException(_T("File is not opened."));
  }
  return feof((FILE *)file_) && bufferCursor_ == bufferContentLen_;
}

void asio::BinaryReader::ReadLine( axis::String& s )
{
  bool hasReachedEndOfLine = false;
  s.clear();
  unsigned int c[4];

  while (!hasReachedEndOfLine && !IsEOF())
  {
    int bytesRead = 1;
    c[0] = buffer_[bufferCursor_];
    unsigned int codepoint = 0;
    if (c[0] < 0x80)
    { // basic character
      codepoint = c[0];
    }
    else if (c[0] < 0xE0)
    { // 2-byte character
      bufferCursor_++;
      if (bufferCursor_ > bufferContentLen_)
      {
        FillBuffer();
      }
      c[1] = buffer_[bufferCursor_];
      codepoint = c[1] & 0x3F | ((c[0] & 0x1F) << 6);
      bytesRead = 2;
    }
    else if (c[0] < 0xF0)
    { // 3-byte character
      for (int i = 1; i < 3; i++)
      {
        bufferCursor_++;
        if (bufferCursor_ > bufferContentLen_)
        {
          FillBuffer();
        }
        c[i] = buffer_[bufferCursor_];
      }
      codepoint = c[2] & 0x3F | ((c[1] & 0x3F) << 6) | ((c[0] & 0xF) << 12);
      bytesRead = 3;
    }
    else if (c[0] < 0xF8)
    { // 4-byte character
      for (int i = 1; i < 4; i++)
      {
        bufferCursor_++;
        if (bufferCursor_ > bufferContentLen_)
        {
          FillBuffer();
        }
        c[i] = buffer_[bufferCursor_];
      }
      codepoint = c[3] & 0x3F | ((c[2] & 0x3F) << 6) | ((c[1] & 0x3F) << 12) | ((c[0] & 0x7) << 18);
      bytesRead = 4;
    }
    else if (c[0] < 0xFC)
    { // 5-byte character
      for (int i = 1; i < 5; i++)
      {
        bufferCursor_++;
        if (bufferCursor_ > bufferContentLen_)
        {
          FillBuffer();
        }
        c[i] = buffer_[bufferCursor_];
      }
      codepoint = c[4] & 0x3F | ((c[3] & 0x3F) << 6) | ((c[2] & 0x3F) << 12) | ((c[1] & 0x3F) << 18) | ((c[0] & 0x3) << 24);
      bytesRead = 5;
    }
    else
    { // 6-byte character
      for (int i = 1; i < 5; i++)
      {
        bufferCursor_++;
        if (bufferCursor_ > bufferContentLen_)
        {
          FillBuffer();
        }
        c[i] = buffer_[bufferCursor_];
      }
      codepoint = c[5] & 0x3F | ((c[4] & 0x3F) << 6) | ((c[3] & 0x3F) << 12) | ((c[2] & 0x3F) << 18) | ((c[1] & 0x3F) << 24) | ((c[0] & 0x1) << 30);
      bytesRead = 6;
    }
    bufferCursor_++;

    hasReachedEndOfLine = (codepoint == eol);
    if (!hasReachedEndOfLine)
    { // transform to UTF-16
      if (codepoint < 0x10000)
      {
        s.append(codepoint);
      }
      else if (codepoint >= 0x10000)
      {
        unsigned int baseCode = codepoint - 0x10000;
        unsigned int leadSurrogate = ((codepoint & 0xFFC00) >> 10) + 0xD800;
        unsigned int trailSurrogate = (codepoint & 0x3FF) + 0xDC00;
        s.append(leadSurrogate).append(trailSurrogate);
      }

      if (IsEOF())
      {
        curLineNumber_++;
      }
    }
    else
    {
      curLineNumber_++;
    }
    if (bufferCursor_ >= bufferContentLen_)
    {
      FillBuffer();
    }
  } 
}

void asio::BinaryReader::PushBackLine( void )
{
  // fill buffer if we have already read the entire contents
  if (bufferCursor_ == bufferContentLen_)
  {
    FillBuffer();
  }
}

void asio::BinaryReader::Reset( void )
{
  // go back to beginning of file
  if (!fseek((FILE *)file_, 0, SEEK_SET))
  {
    throw axis::foundation::IOException(_T("Could not read from beginning of file."));
  }
  FillBuffer();
  curLineNumber_ = 0;
}

void axis::services::io::BinaryReader::FillBuffer( void )
{
  if (feof((FILE *)file_))
  { 
    bufferContentLen_ = 0;
    bufferCursor_ = 0;
    return;
  }

  uint64 readCount = fread(buffer_, sizeof(char), bufferSize_, (FILE *)file_);
  bufferContentLen_ = readCount;
  bufferCursor_ = 0;
  if (readCount < bufferSize_) 
  { // might indicate a read error
    if (ferror((FILE *)file_))
    {
      throw axis::foundation::IOException(_T("File read operation failed."));
    }
  }
}
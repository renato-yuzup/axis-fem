#include "FileStreamWriter.hpp"
#include "FileStreamWriterPimpl.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/IOException.hpp"

#define WRITE_BUFFER_LENGTH     4096*1024

axis::services::io::FileStreamWriter::FileStreamWriter( const axis::String& fileName )
{
  pimpl_ = new FileStreamWriterPimpl();
  pimpl_->fileName = fileName;
}

axis::services::io::FileStreamWriter::~FileStreamWriter( void )
{
  if (pimpl_->stream.is_open())
  {
    pimpl_->stream.close();
  }
  delete pimpl_;
}

void axis::services::io::FileStreamWriter::Destroy( void ) const
{
  delete this;
}

void axis::services::io::FileStreamWriter::Write( const axis::String& s )
{
  pimpl_->stream << s.data();
  if (pimpl_->stream.fail())
  {	// operation failed
    pimpl_->stream.clear();
    throw axis::foundation::IOException();
  }
}

void axis::services::io::FileStreamWriter::Close( void )
{
  if (pimpl_->stream.is_open())
  {
    pimpl_->stream.close();
  }
  else
  {	// cannot close again
    throw axis::foundation::InvalidOperationException();
  }
}

void axis::services::io::FileStreamWriter::Open( WriteMode writeMode, LockMode lockMode )
{
  std::ios_base::openmode openMode = std::ios_base::out;
  int protectionMode;

  if (writeMode == Append)
  {
    openMode |= std::ios_base::app | std::ios_base::ate;
  }
  else
  {
    openMode |= std::ios_base::trunc;
  }
  if (lockMode == ExclusiveMode)
  {	// deny any shared access
    protectionMode = _SH_DENYRW;
  }
  else
  {	// deny none
    protectionMode = _SH_DENYWR;
  }
  pimpl_->stream.open(pimpl_->fileName.data(), openMode, protectionMode);
  if (pimpl_->stream.fail())
  {	// open failed
    pimpl_->stream.clear();
    throw axis::foundation::IOException();
  }
}

void axis::services::io::FileStreamWriter::Flush( void )
{
  pimpl_->stream.flush();
}


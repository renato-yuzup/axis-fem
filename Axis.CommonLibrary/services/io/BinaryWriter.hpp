#pragma once
#include "StreamWriter.hpp"
#include "foundation/basic_types.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace services { namespace io {

class AXISCOMMONLIBRARY_API BinaryWriter : public StreamWriter
{
public:
  static BinaryWriter& Create(const axis::String& fileName);
  static BinaryWriter& Create(const axis::String& fileName, int bufferLength);

  virtual ~BinaryWriter(void);

  virtual void WriteLine( const axis::String& s );

  virtual void WriteLine( void );

  virtual void Write( const axis::String& s );

  virtual axis::String GetEndOfLineSequence( void ) const;

  virtual void SetEndOfLineSequence( const axis::String& eol );

  virtual unsigned long GetBytesWritten( void ) const;

  virtual bool IsAutoFlush( void ) const;

  virtual bool IsBuffered( void ) const;

  virtual unsigned long GetBufferSize( void ) const;

  virtual unsigned long GetBufferUsedSpace( void ) const;

  virtual bool IsOpen( void ) const;

  virtual void Open( WriteMode writeMode = kOverwrite, LockMode lockMode = kSharedMode );

  virtual void Flush( void );

  virtual void ToggleFlush( void );

  virtual void Close( void );

  virtual axis::String GetStreamPath( void ) const;

  virtual void Destroy( void ) const;
private:
  BinaryWriter(const axis::String& fileName);
  BinaryWriter(const axis::String& fileName, int bufferLength);
  void Init(const axis::String& fileName, int bufferLength);

  void *file_;
  char *buffer_;
  uint64 bufferSize_;
  uint64 bufferCursor_;
  uint64 bytesWritten_;
  const axis::String fileName_;
  axis::String eol_;
  bool isOpened_;
};

} } } // namespace axis::services::io

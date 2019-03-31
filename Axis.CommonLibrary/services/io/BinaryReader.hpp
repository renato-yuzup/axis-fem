#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "StreamReader.hpp"

namespace axis { namespace services { namespace io {

class AXISCOMMONLIBRARY_API BinaryReader : public StreamReader
{
public:
  static BinaryReader& Create(const axis::String& filename);
  static BinaryReader& Create(const axis::String& filename, int bufferLength);
  virtual ~BinaryReader(void);
  virtual void Destroy(void);

  virtual void ReadLine( axis::String& s );

  virtual void PushBackLine( void );

  virtual bool IsEOF( void ) const;

  virtual bool IsOpen( void ) const;

  virtual void Close( void );

  virtual axis::String GetStreamPath( void ) const;

  virtual unsigned long GetLastLineNumber( void ) const;

  virtual void Reset( void );
private:
  BinaryReader(const axis::String& filename);
  BinaryReader(const axis::String& filename, int bufferLength);
  void Init(int bufferLength);
  void FillBuffer(void);

  void *file_;
  axis::String filename_;
  char *buffer_;
  uint64 bufferContentLen_;
  uint64 bufferSize_;
  uint64 bufferCursor_;
  uint64 curLineNumber_;
  bool isOpened_;
};

} } } // namespace axis::services::io

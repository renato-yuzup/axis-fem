#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "StreamWriter.hpp"
#include "FileStreamWriter.hpp"

namespace axis
{
	namespace services
	{
		namespace io
		{
			class AXISCOMMONLIBRARY_API FileWriter : public StreamWriter
			{
			private:
				FileStreamWriter *_stream;

				const axis::String _fileName;
				unsigned long _bytesWritten;
				axis::String _eol;
				bool _isOpened;

				void Init(void);

				FileWriter(const axis::String& fileName);
			public:
				static FileWriter& Create(const axis::String& fileName);

				virtual ~FileWriter(void);

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

				virtual void Flush( void );

				virtual void ToggleFlush( void );

				virtual void Close( void );

				virtual axis::String GetStreamPath( void ) const;

				virtual void Open( WriteMode writeMode = kOverwrite, LockMode lockMode = kSharedMode );

				virtual void Destroy( void ) const;				
			};		
		}
	}
}


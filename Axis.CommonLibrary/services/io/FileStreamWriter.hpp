#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace io
		{
			class AXISCOMMONLIBRARY_API FileStreamWriter
			{
			public:
				enum WriteMode
				{
					Overwrite,
					Append
				};

				enum LockMode
				{
					ExclusiveMode,
					SharedMode
				};

				FileStreamWriter(const axis::String& fileName);
				~FileStreamWriter(void);

        void Destroy( void ) const;
        void Write( const axis::String& s );
        void Close( void );
        void Open(WriteMode writeMode, LockMode lockMode);
        void Flush(void);
      private:
        class FileStreamWriterPimpl;
        FileStreamWriterPimpl *pimpl_;
			};
		}
	}
}


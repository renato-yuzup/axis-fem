#pragma once
#include "DirectoryNavigator.hpp"
#include "boost/filesystem.hpp"

// After version 1.49 of Boost, the filesystem3 is deprecated
// in favor of the filesystem namespace.
#if BOOST_VERSION > 104900
#define AXIS_BOOST_FILESYSTEM	boost::filesystem
#else
#define AXIS_BOOST_FILESYSTEM	boost::filesystem3
#endif

namespace axis
{
	namespace services
	{
		namespace io
		{
			class BoostSingleDirectoryNavigator : public DirectoryNavigator
			{
			private:
				AXIS_BOOST_FILESYSTEM::directory_iterator _end;
				AXIS_BOOST_FILESYSTEM::directory_iterator *_current;
			public:
				BoostSingleDirectoryNavigator(const axis::String& path);
				virtual ~BoostSingleDirectoryNavigator(void);

				virtual void Destroy( void ) const;

				virtual axis::String GetFile( void ) const;

				virtual bool HasNext( void ) const;

				virtual void GoNext( void ) const;

				virtual axis::String GetFileName( void ) const;

				virtual axis::String GetFileExtension( void ) const;
			};
		}
	}
}


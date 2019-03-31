#pragma once
#include "DirectoryNavigator.hpp"
#include <boost/filesystem.hpp>

// After version 1.49 of Boost, the filesystem3 is deprecated
// in favor of the filesystem namespace.
#if BOOST_VERSION > 104900
	#define AXIS_BOOST_FILESYSTEM	boost::filesystem
#else
	#define AXIS_BOOST_FILESYSTEM	boost::filesystem3
#endif

#ifdef _UNICODE
	typedef AXIS_BOOST_FILESYSTEM::wrecursive_directory_iterator dir_iterator;
#else
	typedef AXIS_BOOST_FILESYSTEM::recursive_directory_iterator dir_iterator;
#endif

namespace axis
{
	namespace services
	{
		namespace io
		{
			class BoostRecursiveDirectoryNavigator : public DirectoryNavigator
			{
			private:
				dir_iterator _end;
				dir_iterator *_current;
			public:
				BoostRecursiveDirectoryNavigator(const axis::String& path);
				virtual ~BoostRecursiveDirectoryNavigator(void);

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


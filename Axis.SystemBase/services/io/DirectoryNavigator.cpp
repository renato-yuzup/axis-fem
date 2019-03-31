#include "DirectoryNavigator.hpp"
#include "BoostRecursiveDirectoryNavigator.hpp"
#include "BoostSingleDirectoryNavigator.hpp"

axis::services::io::DirectoryNavigator::DirectoryNavigator( void )
{
	// nothing to do here
}

axis::services::io::DirectoryNavigator::~DirectoryNavigator( void )
{
	// nothing to do here
}

axis::services::io::DirectoryNavigator& axis::services::io::DirectoryNavigator::Create( const axis::String& path, bool isRecursive )
{
	if (!isRecursive)
	{
		return *new BoostSingleDirectoryNavigator(path);
	}
	return *new BoostRecursiveDirectoryNavigator(path);
}

axis::services::io::DirectoryNavigator& axis::services::io::DirectoryNavigator::Create( const axis::String& path )
{
	return Create(path, false);
}


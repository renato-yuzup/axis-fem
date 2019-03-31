#include "BoostSingleDirectoryNavigator.hpp"
#include "foundation/InvalidOperationException.hpp"

axis::services::io::BoostSingleDirectoryNavigator::BoostSingleDirectoryNavigator( const axis::String& path )
{
	_current = new AXIS_BOOST_FILESYSTEM::directory_iterator(path.c_str());
}

axis::services::io::BoostSingleDirectoryNavigator::~BoostSingleDirectoryNavigator( void )
{
	delete _current;
}

void axis::services::io::BoostSingleDirectoryNavigator::Destroy( void ) const
{
	delete this;
}

axis::String axis::services::io::BoostSingleDirectoryNavigator::GetFile( void ) const
{
	if (*_current == _end)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return (*_current)->path().c_str();
}

bool axis::services::io::BoostSingleDirectoryNavigator::HasNext( void ) const
{
	return *_current != _end;
}

void axis::services::io::BoostSingleDirectoryNavigator::GoNext( void ) const
{
	if (!HasNext())
	{
		throw axis::foundation::InvalidOperationException();
	}
	++(*_current);
}

axis::String axis::services::io::BoostSingleDirectoryNavigator::GetFileName( void ) const
{
	if (*_current == _end)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return (*_current)->path().filename().c_str();
}

axis::String axis::services::io::BoostSingleDirectoryNavigator::GetFileExtension( void ) const
{
	if (*_current == _end)
	{
		throw axis::foundation::InvalidOperationException();
	}
	if ((*_current)->path().has_extension())
	{
		return (*_current)->path().extension().c_str();
	}

	// return a null extension
	return _T("");
}


#include "BoostRecursiveDirectoryNavigator.hpp"
#include "foundation/InvalidOperationException.hpp"

axis::services::io::BoostRecursiveDirectoryNavigator::BoostRecursiveDirectoryNavigator( const axis::String& path )
{
	_current = new dir_iterator(path.c_str());
}

axis::services::io::BoostRecursiveDirectoryNavigator::~BoostRecursiveDirectoryNavigator( void )
{
	delete _current;
}

void axis::services::io::BoostRecursiveDirectoryNavigator::Destroy( void ) const
{
	delete this;
}

axis::String axis::services::io::BoostRecursiveDirectoryNavigator::GetFile( void ) const
{
	if (*_current == _end)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return (*_current)->path().c_str();
}

bool axis::services::io::BoostRecursiveDirectoryNavigator::HasNext( void ) const
{
	return (*_current) != _end;
}

void axis::services::io::BoostRecursiveDirectoryNavigator::GoNext( void ) const
{
	if (!HasNext())
	{
		throw axis::foundation::InvalidOperationException();
	}
	++(*_current);
}

axis::String axis::services::io::BoostRecursiveDirectoryNavigator::GetFileName( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

axis::String axis::services::io::BoostRecursiveDirectoryNavigator::GetFileExtension( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}


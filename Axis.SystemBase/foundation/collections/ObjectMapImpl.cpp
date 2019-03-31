#include "ObjectMapImpl.hpp"
#include "foundation/ArgumentException.hpp"

axis::foundation::collections::ObjectMapImpl::~ObjectMapImpl( void )
{
	// nothing to do here
}

void axis::foundation::collections::ObjectMapImpl::Destroy( void ) const
{
	delete this;
}

void axis::foundation::collections::ObjectMapImpl::Add( const axis::String& id, Collectible& obj )
{
	if (Contains(id))
	{
		throw axis::foundation::ArgumentException();
	}
	_objects[id] = &obj;
}

void axis::foundation::collections::ObjectMapImpl::Remove( const axis::String& id )
{
	if (!Contains(id))
	{
		throw axis::foundation::ArgumentException();
	}
	_objects.erase(id);
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectMapImpl::Get( const axis::String& id ) const
{
	return operator [](id);
}

bool axis::foundation::collections::ObjectMapImpl::Contains( const axis::String& id ) const
{
	return _objects.find(id) != _objects.end();
}

void axis::foundation::collections::ObjectMapImpl::Clear( void )
{
	_objects.clear();
}

size_type axis::foundation::collections::ObjectMapImpl::Count( void ) const
{
	return (size_type)_objects.size();
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectMapImpl::operator[]( const axis::String& id ) const
{
	if (!Contains(id))
	{
		throw axis::foundation::ArgumentException();
	}
	return *_objects.at(id);
}

void axis::foundation::collections::ObjectMapImpl::DestroyChildren( void )
{
	collection::iterator end = _objects.end();
	for(collection::iterator it = _objects.begin(); it != end; ++it)
	{
		delete it->second;
	}
	Clear();
}

axis::foundation::collections::ObjectMap::IteratorLogic& axis::foundation::collections::ObjectMapImpl::DoGetIterator( void ) const
{
	return *new axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl(_objects.begin(), _objects.end());
}

axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::IteratorLogicImpl( const collection::const_iterator& current, const collection::const_iterator& end )
{
	_current = current;
	_end = end;
}

void axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::Destroy( void ) const
{
	delete this;
}

bool axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::HasNext( void ) const
{
	return _current != _end;
}

void axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::GoNext( void )
{
	++_current;
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::GetItem( void ) const
{
	return *(_current->second);
}

axis::String axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::GetKey( void ) const
{
	return _current->first;
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::operator*( void ) const
{
	return *(_current->second);
}

axis::foundation::collections::ObjectMap::IteratorLogic& axis::foundation::collections::ObjectMapImpl::IteratorLogicImpl::Clone( void ) const
{
	return *new IteratorLogicImpl(_current, _end);
}

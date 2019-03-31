#include "ObjectSetImpl.hpp"
#include "foundation\ArgumentException.hpp"

axis::foundation::collections::ObjectSetImpl::ObjectSetImpl( void )
{
	// nothing to do here
}

axis::foundation::collections::ObjectSetImpl::~ObjectSetImpl( void )
{
	// nothing to do here
}

void axis::foundation::collections::ObjectSetImpl::Destroy( void ) const
{
	delete this;
}

void axis::foundation::collections::ObjectSetImpl::Add( Collectible& object )
{
	if (Contains(object))
	{
		throw axis::foundation::ArgumentException();
	}
	_objects.insert(&object);
}

bool axis::foundation::collections::ObjectSetImpl::Contains( const Collectible& object ) const
{
	return _objects.find(const_cast<Collectible *>(&object)) != _objects.cend();
}

void axis::foundation::collections::ObjectSetImpl::Remove( const Collectible& object )
{
	if (!Contains(object))
	{
		throw axis::foundation::ArgumentException();
	}
	_objects.erase(const_cast<Collectible *>(&object));
}

void axis::foundation::collections::ObjectSetImpl::Clear( void )
{
	_objects.clear();
}

size_type axis::foundation::collections::ObjectSetImpl::Count( void ) const
{
	return (size_type)_objects.size();
}

axis::foundation::collections::ObjectSet::IteratorLogic& axis::foundation::collections::ObjectSetImpl::DoGetIterator( void ) const
{
	return *new axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl(_objects.begin(), _objects.end());
}

axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl::IteratorLogicImpl( const list::iterator& current, const list::iterator& end )
{
	_current = current;
	_end = end;
}

void axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl::Destroy( void ) const
{
	delete this;
}

bool axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl::HasNext( void ) const
{
	return _current != _end;
}

void axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl::GoNext( void )
{
	++_current;
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl::GetItem( void ) const
{
	return **_current;
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl::operator*( void ) const
{
	return **_current;
}

axis::foundation::collections::ObjectSet::IteratorLogic& axis::foundation::collections::ObjectSetImpl::IteratorLogicImpl::Clone( void ) const
{
	return *new IteratorLogicImpl(_current, _end);
}
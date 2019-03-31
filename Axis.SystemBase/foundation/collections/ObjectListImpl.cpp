#include "ObjectListImpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/ElementNotFoundException.hpp"

axis::foundation::collections::ObjectListImpl::ObjectListImpl( void )
{
	// nothing to do here
}

axis::foundation::collections::ObjectListImpl::~ObjectListImpl( void )
{
	// nothing to do here
}

void axis::foundation::collections::ObjectListImpl::Destroy( void ) const
{
	delete this;
}

void axis::foundation::collections::ObjectListImpl::Add( Collectible& object )
{
	_objects.push_back(&object);
}

bool axis::foundation::collections::ObjectListImpl::Contains( Collectible& object ) const
{
	list::const_iterator it = _objects.begin();
	while (*it != &object) ++it;
	return it != _objects.end();
}

void axis::foundation::collections::ObjectListImpl::Remove( Collectible& object )
{
	// search for the element
	list::const_iterator it = _objects.begin();
	while (*it != &object) ++it;

	if (it == _objects.end())
	{	// couldn't find element
		throw axis::foundation::ArgumentException();
	}
	_objects.erase(it);
}

void axis::foundation::collections::ObjectListImpl::Clear( void )
{
	_objects.clear();
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectListImpl::Get( size_t index ) const
{
	return operator [](index);
}

size_t axis::foundation::collections::ObjectListImpl::GetIndex( Collectible& item ) const
{
	size_t count = _objects.size();
	for (size_t i = 0; i < count; ++i)
	{
		if (_objects[i] == &item)
		{
			return i;
		}
	}

	// element not found
	throw axis::foundation::ElementNotFoundException();
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectListImpl::operator[]( size_t index ) const
{
	if (index > Count() - 1)
	{
		throw axis::foundation::OutOfBoundsException();
	}

	return *(_objects[index]);
}

size_type axis::foundation::collections::ObjectListImpl::Count( void ) const
{
	return (size_type)_objects.size();
}

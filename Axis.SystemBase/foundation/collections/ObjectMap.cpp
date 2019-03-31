#include "ObjectMap.hpp"
#include "ObjectMapImpl.hpp"

axis::foundation::collections::ObjectMap::~ObjectMap( void )
{
	// nothing to do here
}

axis::foundation::collections::ObjectMap& axis::foundation::collections::ObjectMap::Create( void )
{
	return *new ObjectMapImpl();
}

axis::foundation::collections::ObjectMap::Iterator axis::foundation::collections::ObjectMap::GetIterator( void ) const
{
	return Iterator(DoGetIterator());
}

bool axis::foundation::collections::ObjectMap::Iterator::HasNext( void ) const
{
	return _logic->HasNext();
}

void axis::foundation::collections::ObjectMap::Iterator::GoNext( void )
{
	_logic->GoNext();
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectMap::Iterator::GetItem( void ) const
{
	return _logic->GetItem();
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectMap::Iterator::operator*( void ) const
{
	return _logic->GetItem();
}

axis::String axis::foundation::collections::ObjectMap::Iterator::GetKey( void ) const
{
	return _logic->GetKey();
}

axis::foundation::collections::ObjectMap::Iterator::Iterator( IteratorLogic& logic )
{
	_logic = &logic;
}

axis::foundation::collections::ObjectMap::Iterator::Iterator( const Iterator& other )
{
	_logic = &other._logic->Clone();
}

axis::foundation::collections::ObjectMap::Iterator& axis::foundation::collections::ObjectMap::Iterator::operator=( const Iterator& other )
{
	// avoid self assignment
	if (this == &other) return *this;

	_logic->Destroy();
	_logic = &other._logic->Clone();
	return *this;
}

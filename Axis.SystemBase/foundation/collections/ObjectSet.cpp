#include "ObjectSet.hpp"
#include "ObjectSetImpl.hpp"

axis::foundation::collections::ObjectSet::~ObjectSet( void )
{
	// nothing to do here
}

axis::foundation::collections::ObjectSet& axis::foundation::collections::ObjectSet::Create( void )
{
	return *new ObjectSetImpl();
}

axis::foundation::collections::ObjectSet::Iterator axis::foundation::collections::ObjectSet::GetIterator( void ) const
{
	return Iterator(DoGetIterator());
}

bool axis::foundation::collections::ObjectSet::Iterator::HasNext( void ) const
{
	return _logic->HasNext();
}

void axis::foundation::collections::ObjectSet::Iterator::GoNext( void )
{
	_logic->GoNext();
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectSet::Iterator::GetItem( void ) const
{
	return _logic->GetItem();
}

axis::foundation::collections::Collectible& axis::foundation::collections::ObjectSet::Iterator::operator*( void ) const
{
	return _logic->GetItem();
}

axis::foundation::collections::ObjectSet::Iterator::Iterator( IteratorLogic& logic )
{
	_logic = &logic;
}

axis::foundation::collections::ObjectSet::Iterator::Iterator( const Iterator& other )
{
	_logic = &other._logic->Clone();
}

axis::foundation::collections::ObjectSet::Iterator& axis::foundation::collections::ObjectSet::Iterator::operator=( const Iterator& other )
{
	// avoid self assignment
	if (this == &other) return *this;

	_logic->Destroy();
	_logic = &other._logic->Clone();
	return *this;
}
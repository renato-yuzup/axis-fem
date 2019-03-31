#include "AssociationSet.hpp"
#include "AssociationSetImpl.hpp"

axis::foundation::collections::AssociationSet::~AssociationSet( void )
{
	// nothing to do here
}

axis::foundation::collections::AssociationSet& axis::foundation::collections::AssociationSet::Create( void )
{
	return *new AssociationSetImpl();
}

axis::foundation::collections::AssociationSet::Iterator axis::foundation::collections::AssociationSet::GetIterator( void ) const
{
	return Iterator(DoGetIterator());
}

bool axis::foundation::collections::AssociationSet::Iterator::HasNext( void ) const
{
	return _logic->HasNext();
}

void axis::foundation::collections::AssociationSet::Iterator::GoNext( void )
{
	_logic->GoNext();
}

axis::foundation::collections::AssociationSet::value_type& axis::foundation::collections::AssociationSet::Iterator::GetItem( void ) const
{
	return _logic->GetItem();
}

axis::foundation::collections::Collectible& axis::foundation::collections::AssociationSet::Iterator::operator*( void ) const
{
	return _logic->GetItem();
}

axis::foundation::collections::AssociationSet::key_type axis::foundation::collections::AssociationSet::Iterator::GetKey( void ) const
{
	return _logic->GetKey();
}

axis::foundation::collections::AssociationSet::Iterator::Iterator( IteratorLogic& logic )
{
	_logic = &logic;
}

axis::foundation::collections::AssociationSet::Iterator::Iterator( const Iterator& other )
{
	_logic = &other._logic->Clone();
}

axis::foundation::collections::AssociationSet::Iterator& axis::foundation::collections::AssociationSet::Iterator::operator=( const Iterator& other )
{
	// avoid self assignment
	if (this == &other) return *this;

	_logic->Destroy();
	_logic = &other._logic->Clone();
	return *this;
}

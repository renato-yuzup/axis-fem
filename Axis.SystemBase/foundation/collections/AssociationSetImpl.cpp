#include "AssociationSetImpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"

axis::foundation::collections::AssociationSetImpl::~AssociationSetImpl( void )
{
	// nothing to do here
}

void axis::foundation::collections::AssociationSetImpl::Destroy( void ) const
{
	delete this;
}

void axis::foundation::collections::AssociationSetImpl::Add( const_key_type id, value_type& obj )
{
	if (Contains(id))
	{
		throw axis::foundation::ArgumentException();
	}
	collection::nth_index<0>::type& set_index = _objects.get<0>();
	set_index.insert(mutable_pair(id, &obj));
}

void axis::foundation::collections::AssociationSetImpl::Remove( const_key_type id )
{
	if (!Contains(id))
	{
		throw axis::foundation::ArgumentException();
	}
	collection::nth_index<0>::type& set_index = _objects.get<0>();
	set_index.erase(id);
}

axis::foundation::collections::AssociationSet::value_type& axis::foundation::collections::AssociationSetImpl::Get( const_key_type id ) const
{
	return operator [](id);
}

axis::foundation::collections::AssociationSet::value_type& axis::foundation::collections::AssociationSetImpl::Get( size_type id ) const
{
	return operator [](id);
}

axis::foundation::collections::AssociationSet::key_type axis::foundation::collections::AssociationSetImpl::GetKey( size_type id ) const
{
	if (id >= Count())
	{
		throw axis::foundation::OutOfBoundsException();
	}
	const collection::nth_index<1>::type& set_index = _objects.get<1>();
	return set_index.at(id).first;
}

bool axis::foundation::collections::AssociationSetImpl::Contains( const_key_type id ) const
{
	const collection::nth_index<0>::type& set_index = _objects.get<0>();
	return set_index.find(id) != set_index.end();
}

void axis::foundation::collections::AssociationSetImpl::Clear( void )
{
	collection::nth_index<0>::type& set_index = _objects.get<0>();
	set_index.clear();
}

size_type axis::foundation::collections::AssociationSetImpl::Count( void ) const
{
	const collection::nth_index<0>::type& set_index = _objects.get<0>();
	return (size_type)set_index.size();
}

axis::foundation::collections::AssociationSet::value_type& axis::foundation::collections::AssociationSetImpl::operator[]( const_key_type id ) const
{
	if (!Contains(id))
	{
		throw axis::foundation::ArgumentException();
	}
	const collection::nth_index<0>::type& set_index = _objects.get<0>();
	return *set_index.find(id)->second;
}

axis::foundation::collections::AssociationSet::value_type& axis::foundation::collections::AssociationSetImpl::operator[]( size_type id ) const
{
	if (id >= Count())
	{
		throw axis::foundation::OutOfBoundsException();
	}
	const collection::nth_index<1>::type& set_index = _objects.get<1>();
	return *set_index.at(id).second;
}

void axis::foundation::collections::AssociationSetImpl::DestroyChildren( void )
{
	const collection::nth_index<1>::type& set_index = _objects.get<1>();
	collection::nth_index<1>::type::iterator it;
	collection::iterator end = _objects.end();
	for(collection::iterator it = _objects.begin(); it != end; ++it)
	{
		delete it->second;
	}
	Clear();
}

axis::foundation::collections::AssociationSet::IteratorLogic& axis::foundation::collections::AssociationSetImpl::DoGetIterator( void ) const
{
	return *new axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl(_objects.begin(), _objects.end());
}

axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::IteratorLogicImpl( const collection::const_iterator& current, const collection::const_iterator& end )
{
	_current = current;
	_end = end;
}

void axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::Destroy( void ) const
{
	delete this;
}

bool axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::HasNext( void ) const
{
	return _current != _end;
}

void axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::GoNext( void )
{
	++_current;
}

axis::foundation::collections::Collectible& axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::GetItem( void ) const
{
	return *(_current->second);
}

axis::foundation::collections::AssociationSet::key_type axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::GetKey( void ) const
{
	return _current->first;
}

axis::foundation::collections::Collectible& axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::operator*( void ) const
{
	return *(_current->second);
}

axis::foundation::collections::AssociationSet::IteratorLogic& axis::foundation::collections::AssociationSetImpl::IteratorLogicImpl::Clone( void ) const
{
	return *new IteratorLogicImpl(_current, _end);
}

axis::foundation::collections::AssociationSetImpl::mutable_pair::mutable_pair( first_type& f,const second_type& s ) : first(f), second(s)
{
	// nothing to do here
}

axis::foundation::collections::AssociationSetImpl::mutable_pair::mutable_pair( void ) : first(NULL), second(NULL)
{
	// nothing to do here
}

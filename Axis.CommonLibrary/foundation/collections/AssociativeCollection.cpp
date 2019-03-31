#include "AssociativeCollection.hpp"
#include "AssociativeCollection_Pimpl.hpp"

using axis::foundation::collections::AssociativeCollection;

template<class Key, class Value>
AssociativeCollection<Key, Value>::AssociativeCollection(void)
{
  pimpl_ = new Pimpl();
}

template<class Key, class Value>
AssociativeCollection<Key, Value>::~AssociativeCollection(void)
{
  delete pimpl_;
  pimpl_ = NULL;
}

template <class Key, class Value>
void AssociativeCollection<Key, Value>::Destroy( void ) const
{
  delete this;
}

template <class Key, class Value>
void AssociativeCollection<Key, Value>::DestroyChildren( void )
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->associations.get<1>();
  for (size_type i = 0; i < index.size(); ++i)
  {
    index[i].second->Destroy();
  }
  index.clear();
}

template <class Key, class Value>
void AssociativeCollection<Key, Value>::Add( KeyType& accessorKey, ValueType& value )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->associations.get<0>();
  index.insert(Pimpl::mutable_pair(&accessorKey, &value));
}

template <class Key, class Value>
bool AssociativeCollection<Key, Value>::Contains( KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->associations.get<0>();
  return index.find(&key) != index.end();
}

template <class Key, class Value>
void AssociativeCollection<Key, Value>::Remove( KeyType& key )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->associations.get<0>();
  index.erase(&key);
}

template <class Key, class Value>
void AssociativeCollection<Key, Value>::Remove( size_type index )
{
  Pimpl::collection::nth_index<0>::type& mapIndex = pimpl_->associations.get<0>();
  Pimpl::collection::nth_index<1>::type& seqIndex = pimpl_->associations.get<1>();
  mapIndex.erase(seqIndex.at(index).first);
}

template <class Key, class Value>
void AssociativeCollection<Key, Value>::Clear( void )
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->associations.get<1>();
  index.clear();
}

template <class Key, class Value>
typename AssociativeCollection<Key, Value>::ValueType& AssociativeCollection<Key, Value>::Get( KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->associations.get<0>();
  return *index.find(&key)->second;
}

template <class Key, class Value>
typename AssociativeCollection<Key, Value>::ValueType& AssociativeCollection<Key, Value>::Get( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->associations.get<1>();
  return *c_index.at(index).second;
}

template <class Key, class Value>
typename AssociativeCollection<Key, Value>::ValueType& AssociativeCollection<Key, Value>::operator[]( KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->associations.get<0>();
  return *index.find(&key)->second;
}

template <class Key, class Value>
typename AssociativeCollection<Key, Value>::ValueType& AssociativeCollection<Key, Value>::operator[]( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->associations.get<1>();
  return *c_index.at(index).second;
}

template <class Key, class Value>
typename AssociativeCollection<Key, Value>::KeyType& AssociativeCollection<Key, Value>::GetKey( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->associations.get<1>();
  return *c_index.at(index).first;
}

template <class Key, class Value>
size_type AssociativeCollection<Key, Value>::Count( void ) const
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->associations.get<1>();
  return (size_type)index.size();
}

template <class Key, class Value>
AssociativeCollection<Key, Value>::Pimpl::mutable_pair::mutable_pair(void)
{
  first = NULL;
  second = NULL;
}

template <class Key, class Value>
AssociativeCollection<Key, Value>::Pimpl::mutable_pair::mutable_pair(first_type f,const second_type s)
{
  first = f;
  second = s;
}

#include "MappedSet.hpp"
#include "MappedSet_Pimpl.hpp"

using axis::foundation::collections::MappedSet;

template<class Key, class Value>
MappedSet<Key, Value>::MappedSet(void)
{
  pimpl_ = new Pimpl();
}

template<class Key, class Value>
MappedSet<Key, Value>::~MappedSet(void)
{
  delete pimpl_;
  pimpl_ = NULL;
}

template<class Key, class Value>
void MappedSet<Key, Value>::Destroy( void ) const
{
  delete this;
}

template<class Key, class Value>
void MappedSet<Key, Value>::DestroyChildren( void )
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->items.get<1>();
  for (size_type i = 0; i < index.size(); ++i)
  {
    index[i].second->Destroy();
  }
  index.clear();
}

template<class Key, class Value>
void MappedSet<Key, Value>::Add( const KeyType& accessorKey, ValueType& value )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  index.insert(Pimpl::mutable_pair(accessorKey, &value));
}

template<class Key, class Value>
bool MappedSet<Key, Value>::Contains( const KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return index.find(&key) != index.end();
}

template<class Key, class Value>
void MappedSet<Key, Value>::Remove( const KeyType& key )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  index.erase(key);
}

template<class Key, class Value>
void MappedSet<Key, Value>::Remove( size_type index )
{
  Pimpl::collection::nth_index<0>::type& mapIndex = pimpl_->items.get<0>();
  Pimpl::collection::nth_index<1>::type& seqIndex = pimpl_->items.get<1>();
  mapIndex.erase(seqIndex.at(index).first);
}

template<class Key, class Value>
void MappedSet<Key, Value>::Clear( void )
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->items.get<1>();
  index.clear();
}

template<class Key, class Value>
typename MappedSet<Key, Value>::ValueType& MappedSet<Key, Value>::Get( const KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return *index.find(key)->second;
}

template<class Key, class Value>
typename MappedSet<Key, Value>::ValueType& MappedSet<Key, Value>::Get( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return *c_index.at(index).second;
}

template <class Key, class Value>
typename MappedSet<Key, Value>::ValueType& MappedSet<Key, Value>::operator[]( const KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return *index.find(key)->second;
}

template <class Key, class Value>
typename MappedSet<Key, Value>::ValueType& MappedSet<Key, Value>::operator[]( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return *c_index.at(index).second;
}

template<class Key, class Value>
typename MappedSet<Key, Value>::KeyType MappedSet<Key, Value>::GetKey( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return c_index.at(index).first;
}

template<class Key, class Value>
size_type MappedSet<Key, Value>::Count( void ) const
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->items.get<1>();
  return (size_type)index.size();
}

template<class Key, class Value>
MappedSet<Key, Value>::Pimpl::mutable_pair::mutable_pair(void)
{
  second = NULL;
}

template<class Key, class Value>
MappedSet<Key, Value>::Pimpl::mutable_pair::mutable_pair(const first_type& f, const second_type s) :
  first(f)
{
  second = s;
}

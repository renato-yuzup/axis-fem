#include "RelativeMap.hpp"
#include "RelativeMap_Pimpl.hpp"
#include "foundation/memory/pointer.hpp"

namespace afc = axis::foundation::collections;
namespace afm = axis::foundation::memory;

template<class Key, class Value>
afc::RelativeMap<Key, Value>::RelativeMap(void)
{
  pimpl_ = new Pimpl();
}

template<class Key, class Value>
afc::RelativeMap<Key, Value>::~RelativeMap(void)
{
  delete pimpl_;
  pimpl_ = NULL;
}

template<class Key, class Value>
void afc::RelativeMap<Key, Value>::Destroy( void ) const
{
  delete this;
}

template<class Key, class Value>
void afc::RelativeMap<Key, Value>::DestroyChildren( void )
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->items.get<1>();
  for (size_type i = 0; i < index.size(); ++i)
  {    
    absref<Value>(index[i].second).Destroy();
    System::ModelMemory().Deallocate(index[i].second);
  }
  index.clear();
}

template<class Key, class Value>
void afc::RelativeMap<Key, Value>::Add( const KeyType& accessorKey, afm::RelativePointer& valuePtr )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  index.insert(Pimpl::mutable_pair(accessorKey, valuePtr));
}

template<class Key, class Value>
bool afc::RelativeMap<Key, Value>::Contains( const KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return index.find(&key) != index.end();
}

template<class Key, class Value>
void afc::RelativeMap<Key, Value>::Remove( const KeyType& key )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  index.erase(key);
}

template<class Key, class Value>
void afc::RelativeMap<Key, Value>::Remove( size_type index )
{
  Pimpl::collection::nth_index<0>::type& mapIndex = pimpl_->items.get<0>();
  Pimpl::collection::nth_index<1>::type& seqIndex = pimpl_->items.get<1>();
  mapIndex.erase(seqIndex.at(index).first);
}

template<class Key, class Value>
void afc::RelativeMap<Key, Value>::Clear( void )
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->items.get<1>();
  index.clear();
}

template<class Key, class Value>
const typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::Get( const KeyType& key ) const
{
  return operator [](key);
}

template<class Key, class Value>
const typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::Get( size_type index ) const
{
  return operator [](index);
}

template<class Key, class Value>
typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::Get( const KeyType& key )
{
  return operator [](key);
}

template<class Key, class Value>
typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::Get( size_type index )
{
  return operator [](index);
}

template <class Key, class Value>
const afm::RelativePointer afc::RelativeMap<Key, Value>::GetPointer( const KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return index.find(key)->second;
}

template <class Key, class Value>
const afm::RelativePointer afc::RelativeMap<Key, Value>::GetPointer( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return c_index.at(index).second;
}

template <class Key, class Value>
afm::RelativePointer afc::RelativeMap<Key, Value>::GetPointer( const KeyType& key )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return index.find(key)->second;
}

template <class Key, class Value>
afm::RelativePointer afc::RelativeMap<Key, Value>::GetPointer( size_type index ) 
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return c_index.at(index).second;
}

template <class Key, class Value>
const typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::operator[]( const KeyType& key ) const
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return absref<Value>(index.find(key)->second);
}

template <class Key, class Value>
const typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::operator[]( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return absref<Value>(c_index.at(index).second);
}

template <class Key, class Value>
typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::operator[]( const KeyType& key )
{
  Pimpl::collection::nth_index<0>::type& index = pimpl_->items.get<0>();
  return absref<Value>(index.find(key)->second);
}

template <class Key, class Value>
typename afc::RelativeMap<Key, Value>::ValueType& afc::RelativeMap<Key, Value>::operator[]( size_type index )
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return absref<Value>(c_index.at(index).second);
}

template<class Key, class Value>
typename afc::RelativeMap<Key, Value>::KeyType afc::RelativeMap<Key, Value>::GetKey( size_type index ) const
{
  Pimpl::collection::nth_index<1>::type& c_index = pimpl_->items.get<1>();
  return c_index.at(index).first;
}

template<class Key, class Value>
size_type afc::RelativeMap<Key, Value>::Count( void ) const
{
  Pimpl::collection::nth_index<1>::type& index = pimpl_->items.get<1>();
  return (size_type)index.size();
}

template<class Key, class Value>
afc::RelativeMap<Key, Value>::Pimpl::mutable_pair::mutable_pair(void)
{
  // nothing to do here
}

template<class Key, class Value>
afc::RelativeMap<Key, Value>::Pimpl::mutable_pair::mutable_pair(const first_type& f, const second_type& s) :
  first(f), second(s)
{
  // nothing to do here
}

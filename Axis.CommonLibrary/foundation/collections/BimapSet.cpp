#include "BimapSet.hpp"
#include "BimapSet_Pimpl.hpp"
#include "foundation/DuplicateKeyException.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/memory/pointer.hpp"

using axis::foundation::collections::BimapSet;
namespace afm = axis::foundation::memory;
namespace afc = axis::foundation::collections;

template <class Value>
BimapSet<Value>::BimapSet( void )
{
  pimpl_ = new Pimpl();
}

template <class Value>
BimapSet<Value>::~BimapSet( void )
{
  delete pimpl_;
}

template <class Value>
void BimapSet<Value>::Add( afm::RelativePointer& item )
{
  Value& val = absref<Value>(item);
  Pimpl::collection::nth_index<1>::type& internalIndex = pimpl_->items.get<1>();
  if (internalIndex.find(val.GetInternalId()) != internalIndex.end())
  {
    throw axis::foundation::DuplicateKeyException();
  }
  internalIndex.insert(absptr<Value>(item));
  pimpl_->refs_.push_back(item);
}

template <class Value>
bool BimapSet<Value>::IsInternalIndexed( key_type internalId ) const
{
  Pimpl::collection::nth_index<1>::type& internalIndex = pimpl_->items.get<1>();
  return internalIndex.find(internalId) != internalIndex.end();
}

template <class Value>
bool BimapSet<Value>::IsUserIndexed( key_type userId ) const
{
  Pimpl::collection::nth_index<2>::type& userIndex = pimpl_->items.get<2>();
  return userIndex.find(userId) != userIndex.end();
}

template <class Value>
void BimapSet<Value>::Remove( value_type& item )
{
  Pimpl::collection::nth_index<0>::type& positionalIndex = pimpl_->items.get<0>();
  positionalIndex.erase(positionalIndex.iterator_to(&item));
}

template <class Value>
void BimapSet<Value>::RemoveByInternalIndex( key_type internalId )
{
  Pimpl::collection::nth_index<1>::type& internalIndex = pimpl_->items.get<1>();
  internalIndex.erase(internalId);
}

template <class Value>
void BimapSet<Value>::RemoveByUserIndex( key_type userId )
{
  Pimpl::collection::nth_index<2>::type& userIndex = pimpl_->items.get<2>();
  userIndex.erase(userId);
}

template <class Value>
void BimapSet<Value>::RemoveByPosition( size_type itemIndex )
{
  Pimpl::collection::nth_index<0>::type& positionalIndex = pimpl_->items.get<0>();
  positionalIndex.erase(positionalIndex.iterator_to(positionalIndex[itemIndex]));
}

template <class Value>
typename const BimapSet<Value>::value_type& BimapSet<Value>::GetByInternalIndex( 
    key_type internalId ) const
{
  Pimpl::collection::nth_index<1>::type& internalIndex = pimpl_->items.get<1>();
  Pimpl::collection::nth_index<1>::type::iterator it = internalIndex.find(internalId);
  if (it == internalIndex.end())
  {
    throw axis::foundation::ElementNotFoundException();
  }
  return **it;
}

template <class Value>
typename const BimapSet<Value>::value_type& BimapSet<Value>::GetByUserIndex( 
    key_type userId ) const
{
  Pimpl::collection::nth_index<2>::type& userIndex = pimpl_->items.get<2>();
  Pimpl::collection::nth_index<2>::type::iterator it = userIndex.find(userId);
  if (it == userIndex.end())
  {
    throw axis::foundation::ElementNotFoundException();
  }
  return **it;
}

template <class Value>
typename const BimapSet<Value>::value_type& BimapSet<Value>::GetByPosition( 
    size_type index ) const
{
  Pimpl::collection::nth_index<0>::type& positionalIndex = pimpl_->items.get<0>();
  if (index < 0 || index >= Count())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return *positionalIndex[index];
}

template <class Value>
typename BimapSet<Value>::value_type& BimapSet<Value>::GetByInternalIndex( 
  key_type internalId )
{
  Pimpl::collection::nth_index<1>::type& internalIndex = pimpl_->items.get<1>();
  Pimpl::collection::nth_index<1>::type::iterator it = internalIndex.find(internalId);
  if (it == internalIndex.end())
  {
    throw axis::foundation::ElementNotFoundException();
  }
  return **it;
}

template <class Value>
typename BimapSet<Value>::value_type& BimapSet<Value>::GetByUserIndex( 
  key_type userId )
{
  Pimpl::collection::nth_index<2>::type& externalIndex = pimpl_->items.get<2>();
  Pimpl::collection::nth_index<2>::type::iterator it = externalIndex.find(userId);
  if (it == externalIndex.end())
  {
    throw axis::foundation::ElementNotFoundException();
  }
  return **it;
}

template <class Value>
typename BimapSet<Value>::value_type& BimapSet<Value>::GetByPosition( 
  size_type index )
{
  Pimpl::collection::nth_index<0>::type& positionalIndex = pimpl_->items.get<0>();
  if (index < 0 || index >= Count())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return *positionalIndex[index];
}

template <class Value>
afm::RelativePointer afc::BimapSet<Value>::GetPointerByPosition( size_type position )
{
  return pimpl_->refs_[position];
}

template <class Value>
afm::RelativePointer afc::BimapSet<Value>::GetPointerByInternalId( key_type internalId )
{
  Pimpl::ref_collection::nth_index<1>::type& internalIndex = pimpl_->refs_.get<1>();
  Pimpl::ref_collection::nth_index<1>::type::iterator it = internalIndex.find(internalId);
  if (it == internalIndex.end())
  {
    throw axis::foundation::ElementNotFoundException();
  }
  return *it;
}

template <class Value>
afm::RelativePointer afc::BimapSet<Value>::GetPointerByUserId( key_type userId )
{
  Pimpl::ref_collection::nth_index<2>::type& internalIndex = pimpl_->refs_.get<2>();
  Pimpl::ref_collection::nth_index<2>::type::iterator it = internalIndex.find(userId);
  if (it == internalIndex.end())
  {
    throw axis::foundation::ElementNotFoundException();
  }
  return *it;
}

template <class Value>
void BimapSet<Value>::Clear( void )
{
  Pimpl::collection::nth_index<0>::type& positionalIndex = pimpl_->items.get<0>();
  positionalIndex.clear();
}

template <class Value>
void BimapSet<Value>::DestroyAll( void )
{
  Pimpl::collection::nth_index<0>::type& positionalIndex = pimpl_->items.get<0>();
  size_type count = Count();
  for (size_type i = 0; i < count; ++i)
  {
    positionalIndex[i]->Destroy();
  }
  positionalIndex.clear();
}

template <class Value>
void BimapSet<Value>::Destroy( void ) const
{
  delete this;
}

template <class Value>
size_type BimapSet<Value>::Count( void ) const
{
  Pimpl::collection::nth_index<0>::type& positionalIndex = pimpl_->items.get<0>();
  return (size_type)positionalIndex.size();
}

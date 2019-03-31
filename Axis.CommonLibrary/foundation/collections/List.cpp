#include "List.hpp"
#include "List_Pimpl.hpp"

using axis::foundation::collections::List;

template <class Value>
List<Value>::List( void )
{
  pimpl_ = new Pimpl();
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
}

template <class Value>
List<Value>::~List( void )
{
  delete pimpl_;
}

template <class Value>
void List<Value>::Add( ValueType& bc )
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  idx.push_back(&bc);
}

template <class Value>
bool axis::foundation::collections::List<Value>::Contains( ValueType& item ) const
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  size_t count = idx.size();
  for (size_t i = 0; i < count; ++i)
  {
    if (idx[i] == &item) return true;
  }
  return false;
}

template <class Value>
typename List<Value>::ValueType& List<Value>::Get( size_type index )
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  return *idx.at(index);
}

template <class Value>
const typename List<Value>::ValueType& List<Value>::Get( size_type index ) const
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  return *idx.at(index);
}

template <class Value>
typename List<Value>::ValueType& List<Value>::operator[]( size_type index )
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  return *idx.at(index);
}

template <class Value>
const typename List<Value>::ValueType& List<Value>::operator[]( size_type index ) const
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  return *idx.at(index);
}

template <class Value>
void List<Value>::Remove( ValueType& bc )
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  idx.remove(&bc);
}

template <class Value>
void List<Value>::Clear( void )
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  idx.clear();
}

template <class Value>
size_type List<Value>::Count( void ) const
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  return (size_type)idx.size();
}

template <class Value>
bool List<Value>::Empty( void ) const
{
  Pimpl::collection::nth_index<0>::type& idx = pimpl_->items.get<0>();
  return idx.empty();
}

template <class Value>
void List<Value>::Destroy( void ) const
{
  delete this;
}

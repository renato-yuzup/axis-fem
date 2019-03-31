#pragma once
#include "AssociativeCollection.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/random_access_index.hpp>

namespace axis { namespace foundation { namespace collections {

template <class Key, class Value>
class AssociativeCollection<Key, Value>::Pimpl
{
public:
  struct mutable_pair
  {
  public:
    typedef KeyType   * first_type;
    typedef ValueType * second_type;

    mutable_pair(void);
    mutable_pair(first_type f, const second_type s);

    first_type first;
    mutable second_type second;
  };

  typedef boost::multi_index::random_access<>  numbered_index;
  typedef boost::multi_index::hashed_unique<boost::multi_index::member<mutable_pair,KeyType *,&mutable_pair::first>> unique_index;
  typedef boost::multi_index::indexed_by<unique_index, numbered_index> index;
  typedef boost::multi_index::multi_index_container<mutable_pair, index> collection;

  collection associations;  
};

} } } // namespace axis::foundation::collections

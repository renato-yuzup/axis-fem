#pragma once
#include "List.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/random_access_index.hpp>

namespace axis { namespace foundation { namespace collections {

template <class Value>
class List<Value>::Pimpl
{
public:
  typedef boost::multi_index::random_access<> numbered_index;
  typedef boost::multi_index::indexed_by<numbered_index> index;
  typedef boost::multi_index::multi_index_container<ValueType *, index> collection;

  collection items;  
};

} } } // namespace axis::foundation::collections

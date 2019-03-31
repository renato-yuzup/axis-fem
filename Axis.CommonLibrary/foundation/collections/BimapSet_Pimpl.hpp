#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include "BimapSet.hpp"
#include "foundation/memory/pointer.hpp"

namespace axis { namespace foundation { namespace collections {

template <class Value>
class BimapSet<Value>::Pimpl
{
public:

  // These define a multi-indexed list, in which elements are indexed by
  // position, internal id and external (user) id
  typedef typename BimapSet<Value>::key_type    item_id_type;
  typedef typename BimapSet<Value>::value_type& value_reference;

  typedef boost::multi_index::random_access<>  numbered_index;  // list index
  typedef boost::multi_index::hashed_unique<
          boost::multi_index::const_mem_fun<Value, item_id_type, &Value::GetInternalId>
      > unique_internal_index;
  typedef boost::multi_index::hashed_unique<
          boost::multi_index::const_mem_fun<Value, item_id_type, &Value::GetUserId>
      > unique_user_index;
  typedef boost::multi_index::indexed_by<numbered_index, 
                                         unique_internal_index, 
                                         unique_user_index
      > index;
  typedef boost::multi_index::multi_index_container<Value *, index> collection;

  // These define a list with items indexed by position
  struct item_internal_key
  {
    typedef item_id_type result_type;
    result_type operator ()(const axis::foundation::memory::RelativePointer& ptr) const
    {
      return absref<typename Value>(ptr).GetInternalId();
    }
  };
  struct item_user_key
  {
    typedef item_id_type result_type;
    result_type operator ()(const axis::foundation::memory::RelativePointer& ptr) const
    {
      return absref<typename Value>(ptr).GetUserId();
    }
  };
  typedef boost::multi_index::hashed_unique<item_internal_key> ptr_internal_index;
  typedef boost::multi_index::hashed_unique<item_user_key> ptr_user_index;
  typedef boost::multi_index::indexed_by<numbered_index, 
            ptr_internal_index, ptr_user_index> ptr_index;
  typedef boost::multi_index::multi_index_container<
      axis::foundation::memory::RelativePointer, ptr_index> ref_collection;

  collection items;
  ref_collection refs_;
};

} } } // namespace axis::foundation::collections


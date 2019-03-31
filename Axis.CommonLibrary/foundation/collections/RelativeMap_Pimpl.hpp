#pragma once
#include "RelativeMap.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/tuple/tuple.hpp>
#include "AxisString.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis{ namespace foundation { namespace collections {

using axis::hash_value;
template struct boost::hash<axis::String>;

template <class T>
struct hash_string
{
  std::size_t operator () (const axis::AxisString<T>& str) const
  {
    return boost::hash_range(str.begin(), str.end());
  }
  std::size_t operator () (const axis::AxisString<T> *str) const
  {
    return boost::hash_range(str->begin(), str->end());
  }
};

template <class T>
struct string_equal_to
{
  bool operator () (const axis::AxisString<T>& s1, const axis::AxisString<T>& s2) const
  {
    return s1 == s2;
  }
  bool operator () (const axis::AxisString<T> *s1, const axis::AxisString<T>& s2) const
  {
    return (*s1) == (s2);
  }
};

template <class Key, class Value>
class RelativeMap<Key, Value>::Pimpl
{
public:
  struct mutable_pair
  {
  public:
    typedef KeyType                                   first_type;
    typedef axis::foundation::memory::RelativePointer second_type;

    mutable_pair(void);
    mutable_pair(const first_type& f, const second_type& s);

    first_type first;
    mutable second_type second;
  };

  typedef boost::multi_index::random_access<>  numbered_index;
  typedef boost::multi_index::hashed_unique<
            boost::multi_index::member<mutable_pair,KeyType, &mutable_pair::first>, 
            hash_string<wchar_t>, string_equal_to<wchar_t>> unique_index;
  typedef boost::multi_index::indexed_by<unique_index, numbered_index> index;
  typedef boost::multi_index::multi_index_container<mutable_pair, index> collection;

  collection items;  
};

} } } // namespace axis::foundation::collections

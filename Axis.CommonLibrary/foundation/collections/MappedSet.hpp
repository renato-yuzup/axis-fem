#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis { namespace foundation { namespace collections {

template <class Key, class Value>
class AXISCOMMONLIBRARY_API MappedSet
{
public:
  typedef typename Key    KeyType;
  typedef typename Value  ValueType;

  MappedSet(void);
  ~MappedSet(void);

  void Destroy(void) const;
  void DestroyChildren(void);

  void Add(const KeyType& accessorKey, ValueType& value);
  bool Contains(const KeyType& key) const;
  void Remove(const KeyType& key);
  void Remove(size_type index);
  void Clear(void);

  ValueType& Get(const KeyType& key) const;
  ValueType& Get(size_type index) const;
  KeyType GetKey(size_type index) const;

  ValueType& operator [](const KeyType& key) const;
  ValueType& operator [](size_type index) const;


  size_type Count(void) const;

private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::foundation::collections


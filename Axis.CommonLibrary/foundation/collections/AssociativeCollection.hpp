#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace foundation { namespace collections {

template <class Key, class Value>
class AXISCOMMONLIBRARY_API AssociativeCollection 
{
public:
  typedef typename Key   KeyType;
  typedef typename Value ValueType;

  AssociativeCollection(void);
  ~AssociativeCollection(void);

  void Destroy(void) const;
  void DestroyChildren(void);

  void Add(KeyType& accessorKey, ValueType& value);
  bool Contains(KeyType& key) const;
  void Remove(KeyType& key);
  void Remove(size_type index);
  void Clear(void);

  ValueType& Get(KeyType& key) const;
  ValueType& Get(size_type index) const;
  ValueType& operator [](KeyType& key) const;
  ValueType& operator [](size_type index) const;

  KeyType& GetKey(size_type index) const;

  size_type Count(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::foundation::collections


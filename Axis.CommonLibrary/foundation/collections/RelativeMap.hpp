#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace foundation { namespace collections {

template <class Key, class Value>
class AXISCOMMONLIBRARY_API RelativeMap
{
public:
  typedef typename Key    KeyType;
  typedef typename Value  ValueType;

  RelativeMap(void);
  ~RelativeMap(void);

  void Destroy(void) const;
  void DestroyChildren(void);

  void Add(const KeyType& accessorKey, axis::foundation::memory::RelativePointer& valuePtr);
  bool Contains(const KeyType& key) const;
  void Remove(const KeyType& key);
  void Remove(size_type index);
  void Clear(void);

  const ValueType& Get(const KeyType& key) const;
  const ValueType& Get(size_type index) const;
  ValueType& Get(const KeyType& key);
  ValueType& Get(size_type index);
  const axis::foundation::memory::RelativePointer GetPointer(const KeyType& key) const;
  const axis::foundation::memory::RelativePointer GetPointer(size_type index) const;
  axis::foundation::memory::RelativePointer GetPointer(const KeyType& key);
  axis::foundation::memory::RelativePointer GetPointer(size_type index);
  KeyType GetKey(size_type index) const;

  const ValueType& operator [](const KeyType& key) const;
  const ValueType& operator [](size_type index) const;
  ValueType& operator [](const KeyType& key);
  ValueType& operator [](size_type index);

  size_type Count(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::foundation::collections

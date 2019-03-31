#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace foundation { namespace collections {

template <class Value>
class AXISCOMMONLIBRARY_API BimapSet
{
public:
  typedef id_type        key_type;
  typedef typename Value value_type;

  BimapSet(void);
  ~BimapSet(void);

  void Add(axis::foundation::memory::RelativePointer& item);
  bool IsInternalIndexed(key_type internalId) const;
  bool IsUserIndexed(key_type userId) const;
  void Remove(value_type& item);
  void RemoveByInternalIndex(key_type internalId);
  void RemoveByUserIndex(key_type userId);
  void RemoveByPosition(size_type itemIndex);
  const value_type& GetByInternalIndex(key_type internalId) const;
  const value_type& GetByUserIndex(key_type userId) const;
  const value_type& GetByPosition(size_type index) const;
  axis::foundation::memory::RelativePointer GetPointerByPosition(size_type position);
  axis::foundation::memory::RelativePointer GetPointerByInternalId(key_type internalId);
  axis::foundation::memory::RelativePointer GetPointerByUserId(key_type userId);
  value_type& GetByInternalIndex(key_type internalId);
  value_type& GetByUserIndex(key_type userId);
  value_type& GetByPosition(size_type index);
  void Clear(void);
  void DestroyAll(void);
  void Destroy(void) const;
  size_type Count(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::foundation::collections

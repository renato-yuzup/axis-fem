#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace foundation { namespace collections {

template <class Value>
class AXISCOMMONLIBRARY_API List
{
public:
  typedef typename Value ValueType;

  List(void);
  ~List(void);

  void Add(ValueType& bc);
  bool Contains(ValueType& item) const;
  ValueType& Get(size_type index);
  const ValueType& Get(size_type index) const;
  ValueType& operator [](size_type index);
  const ValueType& operator [](size_type index) const;

  void Remove(ValueType& bc);
  void Clear(void);
  size_type Count(void) const;
  bool Empty(void) const;
  void Destroy(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::foundation::collections

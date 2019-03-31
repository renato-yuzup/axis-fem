#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "nocopy.hpp"

namespace axis { namespace domain { namespace elements {

class ReverseConnectivityList
{
public:
  ~ReverseConnectivityList(void);
  
  int Count(void) const;
  axis::foundation::memory::RelativePointer GetItem(int index) const;
  void SetItem(int index, const axis::foundation::memory::RelativePointer& ptr);

  static axis::foundation::memory::RelativePointer Create(int maxEntryCount);
private:
  ReverseConnectivityList(int connectivityCount);

  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *, void *);
  int count_;
  axis::foundation::memory::RelativePointer firstItem_;

  DISALLOW_COPY_AND_ASSIGN(ReverseConnectivityList);
};

} } } // namespace axis::domain::elements

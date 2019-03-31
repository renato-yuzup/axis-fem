#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace foundation { namespace memory {

class HeapBlockArena;

class AXISSYSTEMBASE_API RelativePointer
{
public:
  enum SourceArena
  {
    kModelMemory,
    kGlobalMemory
  };

  ~RelativePointer(void);
  RelativePointer(void);
  RelativePointer(const RelativePointer& other);
  RelativePointer& operator =(const RelativePointer& other);

  void *operator *(void);
  const void *operator *(void) const;

  bool operator ==(const RelativePointer& ptr) const;
  bool operator !=(const RelativePointer& ptr) const;

  static const RelativePointer NullPtr;

  static RelativePointer FromAbsolute(void *ptr, SourceArena source);
private:
  RelativePointer(size_t relativeAddress, int chunkIndex, int poolId);

  size_t relativeAddress_;
  int chunkId_;
  int poolId_;

  friend class HeapBlockArena;
};

} } } // namespace axis::foundation::memory

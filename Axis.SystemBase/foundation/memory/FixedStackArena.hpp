#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace foundation { namespace memory {

class AXISSYSTEMBASE_API FixedStackArena
{
public:
  ~FixedStackArena(void);
  static FixedStackArena& Create(uint64 heapSize);
  void Destroy(void) const;

  void *Allocate(int size);
  void Deallocate(const void * const ptr);
  void Obliterate(void);
  uint64 GetFreeSpace(void) const;
  uint64 GetTotalSize(void) const;
  bool ContainsAddress(const void * const ptr) const;
  void *GetBaseAddress(void) const;
private:
  struct MemoryBlock;

  FixedStackArena(uint64 heapSize);
  void Reset(void);

  char *arena_;
  void *nextFreeAddress_;
  MemoryBlock *firstBlock_;
  MemoryBlock *lastBlock_;
  uint64 totalSize_;
  uint64 freeSpace_;
  uint64 blockCount_;

  // disallowed operations
  FixedStackArena(const FixedStackArena&);
  FixedStackArena& operator =(const FixedStackArena&);
};

} } } // namespace axis::foundation::memory

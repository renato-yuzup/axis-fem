#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/fwd/mirroring_fwd.hpp"
#include "foundation/fwd/memory_fwd.hpp"
#include "nocopy.hpp"

namespace axis { namespace foundation { namespace mirroring {

class AXISSYSTEMBASE_API HeapReflector
{
public:
  ~HeapReflector(void);

  static void CloneStructure(MemoryAllocator& allocator, axis::foundation::memory::HeapBlockArena& sourceHeap);
  static void InitializeClone(MemoryReflector& reflector, const axis::foundation::memory::HeapBlockArena& sourceHeap);
  static void Mirror(MemoryReflector& destination, axis::foundation::memory::HeapBlockArena& sourceHeap);
  static void Restore(axis::foundation::memory::HeapBlockArena& destinationHeap, MemoryReflector& source);
private:
  HeapReflector(void);

  static void InitializeMetadata(MemoryReflector& reflector, const axis::foundation::memory::HeapBlockArena& sourceHeap);
  static void InitializeChunkMetadata(MemoryReflector& reflector, const axis::foundation::memory::HeapBlockArena& sourceHeap);

  DISALLOW_COPY_AND_ASSIGN(HeapReflector);
};

} } } // namespace axis::foundation::mirroring

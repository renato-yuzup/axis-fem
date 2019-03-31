#pragma once
#include "HeapBlockArena.hpp"

namespace axis { namespace yuzu { namespace foundation { namespace memory {

struct HeapBlockArena::Chunk
{
public:
  Chunk *Previous;
  Chunk *Next;
  uint64 Size;
  uint64 FreeSpace;
  char *StartingAddress;
  char *NextFreeAddress;
  uint64 AllocationCount;
  int Index;
};

} } } } // namespace axis::yuzu::foundation::memory

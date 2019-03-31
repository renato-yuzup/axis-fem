#pragma once
#include <mutex>
#include "HeapBlockArena.hpp"

namespace axis { namespace foundation { namespace memory {

struct HeapBlockArena::Pimpl
{
public:
  std::mutex Mutex;
};

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

} } } // namespace axis::foundation::memory

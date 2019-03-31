#include "RelativePointer.hpp"
#include <assert.h>
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"

#define UNDEFINED_ARENA_POOL_INDEX    -1
#define STRING_ARENA_POOL_INDEX       0
#define MODEL_ARENA_POOL_INDEX        1
#define GLOBAL_ARENA_POOL_INDEX       2

namespace afm = axis::foundation::memory;

const afm::RelativePointer afm::RelativePointer::NullPtr;

afm::RelativePointer::RelativePointer( void ) :
  relativeAddress_(0), chunkId_(0), poolId_(-1)
{
  // nothing to do here
}

afm::RelativePointer::RelativePointer( size_t relativeAddress, int chunkIndex, int poolId ) :
  relativeAddress_(relativeAddress), chunkId_(chunkIndex), poolId_(poolId)
{
  // nothing to do here
}

afm::RelativePointer::RelativePointer( const RelativePointer& other )
{
  relativeAddress_ = other.relativeAddress_;
  chunkId_ = other.chunkId_;
  poolId_ = other.poolId_;
}

afm::RelativePointer::~RelativePointer( void )
{
  // nothing to do here
}

afm::RelativePointer& afm::RelativePointer::operator =( const RelativePointer& other )
{
  relativeAddress_ = other.relativeAddress_;
  chunkId_ = other.chunkId_;
  poolId_ = other.poolId_;
  return *this;
}

void * afm::RelativePointer::operator *( void )
{
#if !defined(__CUDA_ARCH__) // host code
  switch (poolId_)
  {
  case UNDEFINED_ARENA_POOL_INDEX:
    return nullptr;
  case MODEL_ARENA_POOL_INDEX:
    return reinterpret_cast<void *>(
      reinterpret_cast<size_t>(System::ModelMemory().GetChunkBaseAddress(chunkId_)) +
      relativeAddress_);
  case GLOBAL_ARENA_POOL_INDEX:
    return reinterpret_cast<void *>(
      reinterpret_cast<size_t>(System::GlobalMemory().GetChunkBaseAddress(chunkId_)) +
      relativeAddress_);
  default:
    assert(!"Unknown memory pool source!");
  }
  return nullptr;
#else // device code
  afm::HeapBlockArena& arena = *reinterpret_cast<afm::HeapBlockArena *>(afmk::gpuArena);
  return reinterpret_cast<void *>(
    reinterpret_cast<uint64>(arena.GetChunkBaseAddress(chunkId_)) + relativeAddress_);
#endif
}

const void * afm::RelativePointer::operator *( void ) const
{
#if !defined(__CUDA_ARCH__)
  switch (poolId_)
  {
  case UNDEFINED_ARENA_POOL_INDEX:
    return nullptr;
  case MODEL_ARENA_POOL_INDEX:
    return reinterpret_cast<void *>(
      reinterpret_cast<size_t>(System::ModelMemory().GetChunkBaseAddress(chunkId_)) +
      relativeAddress_);
  case GLOBAL_ARENA_POOL_INDEX:
    return reinterpret_cast<void *>(
      reinterpret_cast<size_t>(System::GlobalMemory().GetChunkBaseAddress(chunkId_)) +
      relativeAddress_);
  default:
    assert(!"Unknown memory pool source!");
  }
  return nullptr;
#else
  afm::HeapBlockArena& arena = *reinterpret_cast<afm::HeapBlockArena *>(afmk::gpuArena);
  return reinterpret_cast<void *>(
    reinterpret_cast<uint64>(arena.GetChunkBaseAddress(chunkId_)) + relativeAddress_);
#endif
}

bool axis::foundation::memory::RelativePointer::operator==( const RelativePointer& ptr ) const
{
  return ptr.chunkId_ == chunkId_ && ptr.poolId_ == poolId_ && ptr.relativeAddress_ == relativeAddress_;
}

bool axis::foundation::memory::RelativePointer::operator!=( const RelativePointer& ptr ) const
{
  return !(*this == ptr);
}

afm::RelativePointer afm::RelativePointer::FromAbsolute( void *ptr, SourceArena source)
{
  afm::HeapBlockArena *heap = nullptr;
  int srcId;
  if (source == kGlobalMemory)
  {
    heap = &System::GlobalMemory();
    srcId = GLOBAL_ARENA_POOL_INDEX;
  }
  else
  {
    heap = &System::ModelMemory(); 
    srcId = MODEL_ARENA_POOL_INDEX;
  }
  size_type count = heap->GetFragmentationCount();
  uint64 myPtr = (uint64)ptr;
  for (int i = 0; i < count; ++i)
  {
    uint64 chunkStartAddr = (uint64)heap->GetChunkBaseAddress(i);
    uint64 chunkSize = heap->GetChunkSize(i);
    uint64 chunkEndAddr = chunkStartAddr + chunkSize - 1;
    if (myPtr >= chunkStartAddr && myPtr <= chunkEndAddr)
    {
      return RelativePointer(myPtr-chunkStartAddr, i, srcId);
    }
  }

  return NullPtr;
}

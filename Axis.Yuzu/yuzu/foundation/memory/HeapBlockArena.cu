#include "HeapBlockArena.hpp"
#include "RelativePointer.hpp"
#include "HeapBlockArena_Pimpl.hpp"

namespace ayfm = axis::yuzu::foundation::memory;

ayfm::HeapBlockArena::HeapBlockArena( void )
{
  // nothing to do here
}

ayfm::HeapBlockArena::~HeapBlockArena( void )
{
  // nothing to do here
}

GPU_ONLY uint64 ayfm::HeapBlockArena::GetMaxContiguousAllocatedFreeSpace( void ) const
{
  uint64 maxSpace = 0;
  Chunk *chunk = metadata_.firstChunk_;
  while (chunk != nullptr)
  {
    if (chunk->FreeSpace > maxSpace)
    {
      maxSpace = chunk->FreeSpace;
    }
    chunk = chunk->Next;
  }
  return maxSpace;
}

GPU_ONLY uint64 ayfm::HeapBlockArena::GetNonContiguousAllocatedFreeSpace( void ) const
{
  uint64 totalSpace = 0;
  Chunk *chunk = metadata_.firstChunk_;
  while (chunk != nullptr)
  {
    totalSpace += chunk->FreeSpace;
    chunk = chunk->Next;
  }
  return totalSpace;
}

GPU_ONLY uint64 ayfm::HeapBlockArena::GetTotalSize( void ) const
{
  return metadata_.totalSize_;
}

GPU_ONLY size_type ayfm::HeapBlockArena::GetFragmentationCount( void ) const
{
  return metadata_.chunkCount_;
}

GPU_ONLY void * ayfm::HeapBlockArena::GetChunkBaseAddress( int chunkIndex ) const
{
  return metadata_.chunks_[chunkIndex]->StartingAddress;
}

GPU_ONLY uint64 ayfm::HeapBlockArena::GetChunkSize( int chunkIndex ) const
{
  return metadata_.chunks_[chunkIndex]->Size;
}

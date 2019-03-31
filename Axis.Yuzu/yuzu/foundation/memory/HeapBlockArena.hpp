#pragma once
#include "yuzu/common/gpu.hpp"

#define MAX_BLOCK_ARENA_CHUNK_SIZE    65536

namespace axis { namespace yuzu { namespace foundation { namespace memory {

class RelativePointer;

/**
 * Implements a thread-safe, fast memory allocator, managing a pre-allocated and 
 * growable portion of memory in which allocations are predominant over de-allocations.
**/
class HeapBlockArena
{
public:
  /**
   * Returns the size, in bytes, of the largest contiguous portion
   * of memory in the arena which has not yet been allocated.
   *
   * @return The maximum contiguous free space.
  **/
  GPU_ONLY uint64 GetMaxContiguousAllocatedFreeSpace(void) const;

  /**
   * Returns, in bytes, the total free space size of the arena.
   *
   * @return The non-contiguous free space.
  **/
  GPU_ONLY uint64 GetNonContiguousAllocatedFreeSpace(void) const;

  /**
   * Returns the total usable space in the arena.
   *
   * @return The total space size.
  **/
  GPU_ONLY uint64 GetTotalSize(void) const;

  /**
   * Returns by how much fragments the arena is divided.
   *
   * @return The fragmentation count.
  **/
  GPU_ONLY size_type GetFragmentationCount(void) const;

  /**
   * Returns the base memory address of a chunk (fragment) of this arena.
   *
   * @param chunkIndex Zero-based index of the chunk.
   *
   * @return The memory address of the chunk.
  **/
  GPU_ONLY void *GetChunkBaseAddress(int chunkIndex) const;

  GPU_ONLY uint64 GetChunkSize(int chunkIndex) const;
private:
  struct Chunk;
  struct Metadata
  {
    void *reserved_;
    int poolId_;
    uint64 initialSize_;
    uint64 chunkSize_;
    uint64 totalSize_;
    uint64 freeSpace_;
    size_type chunkCount_;
    Chunk *firstChunk_;
    Chunk *lastChunk_;  
    Chunk *currentChunk_;
    Chunk *chunks_[MAX_BLOCK_ARENA_CHUNK_SIZE];
    int nextChunkIndex_;
  };

  Metadata metadata_;

  // disallowed operations
  HeapBlockArena(void);
  ~HeapBlockArena(void);
  HeapBlockArena(const HeapBlockArena&);
  HeapBlockArena& operator =(const HeapBlockArena&);
};

} } } } // namespace axis::yuzu::foundation::memory

#pragma once
#include "foundation/Axis.SystemBase.hpp"
#define MAX_BLOCK_ARENA_CHUNK_SIZE    65536

namespace axis { namespace foundation { namespace memory {

/**
 * Implements a thread-safe, fast memory allocator, managing a pre-allocated and 
 * growable portion of memory, organized by blocks and stacks. It is expected that
 * allocations and de-allocations are equally frequent.
**/
class AXISSYSTEMBASE_API HeapStackArena
{
public:

  /**
   * Destructor.
  **/
  ~HeapStackArena(void);

  /**
   * Destroys this object and frees space managed by this arena.
  **/
  void Destroy(void) const;

  /**
   * Creates a new arena.
   *
   * @param initialSize Initial pre-allocated memory size, in bytes.
   * @param chunkSize   Size, in bytes, of the memory increment added when arena becomes full.
   *
   * @return A new memory arena.
  **/
  static HeapStackArena& Create(uint64 initialSize, uint64 chunkSize);

  /**
   * Requests for use a portion of memory.
   *
   * @param size The block size, in bytes.
   *
   * @return A pointer to the allocated block.
  **/
  void * Allocate(uint64 size);

  /**
   * Deallocates a portion of memory in this memory.
   *
   * @param ptr Pointer to the portion of memory.
  **/
  void Deallocate(const void * ptr);

  /**
   * Invalidates any allocation made so far, freeing 
   * the entire arena space for use.
  **/
  void Obliterate(void);

  /**
   * Returns the size, in bytes, of the largest contiguous portion
   * of memory in the arena which has not yet been allocated.
   *
   * @return The maximum contiguous free space.
  **/
  uint64 GetMaxContiguousAllocatedFreeSpace(void) const;

  /**
   * Returns, in bytes, the total free space size of the arena.
   *
   * @return The non-contiguous free space.
  **/
  uint64 GetNonContiguousAllocatedFreeSpace(void) const;

  /**
   * Returns the total usable space in the arena.
   *
   * @return The total space size.
  **/
  uint64 GetTotalSize(void) const;

  /**
   * Returns by how much fragments the arena is divided.
   *
   * @return The fragmentation count.
  **/
  size_type GetFragmentationCount(void) const;

  /**
   * Returns the base memory address of a chunk (fragment) of this arena.
   *
   * @param chunkIndex Zero-based index of the chunk.
   *
   * @return The memory address of the chunk.
  **/
  void *GetChunkBaseAddress(int chunkIndex) const;
private:
  struct Chunk;
  struct Pimpl;

  HeapStackArena(uint64 initialSize, uint64 chunkSize);
  void Init(void);
  Chunk * CreateChunk(uint64 size, Chunk *previous);
  void * AllocateBlock(uint64 size, Chunk *chunk);
  bool IsTailAllocation(const void * ptr, uint64 size, const Chunk *chunk) const;
  void ReclaimChunkFreeSpace(Chunk *chunk, const void *tailAllocationPtr);
  Pimpl *pimpl_;
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

  // disallowed operations
  HeapStackArena(const HeapStackArena&);
  HeapStackArena& operator =(const HeapStackArena&);
};

} } } // namespace axis::foundation::memory

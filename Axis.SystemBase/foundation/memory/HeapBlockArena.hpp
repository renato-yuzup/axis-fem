#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/fwd/mirroring_fwd.hpp" 

#define MAX_BLOCK_ARENA_CHUNK_SIZE    65536

namespace axis { namespace foundation { namespace memory {

class RelativePointer;

/**
 * Implements a thread-safe, fast memory allocator, managing a pre-allocated and 
 * growable portion of memory in which allocations are predominant over de-allocations.
**/
class AXISSYSTEMBASE_API HeapBlockArena
{
public:

  /**
   * Destructor.
  **/
  ~HeapBlockArena(void);

  /**
   * Destroys this object and frees space managed by this arena.
  **/
  void Destroy(void) const;

  /**
   * Creates a new arena.
   *
   * @param initialSize Initial pre-allocated memory size, in bytes.
   * @param chunkSize   Size, in bytes, of the memory increment added when arena becomes full.
   * @param poolId      Identifier for this pool (arena).
   *
   * @return A new memory arena.
  **/
  static HeapBlockArena& Create(uint64 initialSize, uint64 chunkSize, int poolId);

  /**
   * Requests for use a portion of memory.
   *
   * @param size The block size, in bytes.
   *
   * @return A relative pointer to the allocated block.
  **/
  RelativePointer Allocate(uint64 size);

  /**
   * Deallocates a portion of memory in this memory.
   *
   * @param ptr Pointer to the portion of memory.
  **/
  void Deallocate(const RelativePointer& ptr);

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

  uint64 GetChunkSize(int chunkIndex) const;

  static const size_type ChunkOverhead;
  static const size_type AllocationOverhead;
  static const size_type HeapOverhead;
private:
  struct Chunk;
  struct Pimpl;
  struct Metadata
  {
    Pimpl *pimpl_;
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

  HeapBlockArena(uint64 initialSize, uint64 chunkSize, int poolId);
  void Init(void);
  Chunk * CreateChunk(uint64 size, Chunk *previous);
  RelativePointer AllocateBlock(uint64 size, Chunk *chunk);

  Metadata metadata_;

  friend class axis::foundation::mirroring::HeapReflector;

  // disallowed operations
  HeapBlockArena(const HeapBlockArena&);
  HeapBlockArena& operator =(const HeapBlockArena&);
};

} } } // namespace axis::foundation::memory

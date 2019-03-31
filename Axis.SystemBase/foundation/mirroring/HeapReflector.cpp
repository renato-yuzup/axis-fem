#include "HeapReflector.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/HeapBlockArena_Pimpl.hpp"
#include "MemoryAllocator.hpp"
#include "MemoryReflector.hpp"

namespace afmm = axis::foundation::mirroring;
namespace afm = axis::foundation::memory;

afmm::HeapReflector::HeapReflector( void )
{
  // nothing to do here
}

afmm::HeapReflector::~HeapReflector(void)
{
  // nothing to do here
}

void afmm::HeapReflector::CloneStructure( MemoryAllocator& allocator, afm::HeapBlockArena& sourceHeap )
{
  // allocate blocks of equal size
  size_type numBlocks = sourceHeap.GetFragmentationCount();
  for (size_type i = 0; i < numBlocks; i++)
  {
    uint64 blockSize = sourceHeap.GetChunkSize(i);
    blockSize += afm::HeapBlockArena::ChunkOverhead;
    if (i == 0) blockSize += afm::HeapBlockArena::HeapOverhead;
    allocator.Allocate(blockSize);
  }
}

void afmm::HeapReflector::InitializeClone( MemoryReflector& reflector, const afm::HeapBlockArena& sourceHeap )
{
  InitializeMetadata(reflector, sourceHeap);
  InitializeChunkMetadata(reflector, sourceHeap);
}

void afmm::HeapReflector::Mirror( MemoryReflector& destination, afm::HeapBlockArena& sourceHeap )
{
  // copy data contained in every chunk of the heap
  typedef afm::HeapBlockArena::Chunk chunk_t;
  const size_type chunkOverhead = afm::HeapBlockArena::ChunkOverhead;
  const size_type heapOverhead = afm::HeapBlockArena::HeapOverhead;

  int blockCount = destination.GetBlockCount();
  for (int idx = 0; idx < blockCount; idx++)
  {
    void *baseAddress = sourceHeap.GetChunkBaseAddress(idx);
    uint64 chunkSize = sourceHeap.GetChunkSize(idx);
    if (idx == 0)
    {
      destination.WriteToBlock(0, heapOverhead+chunkOverhead, baseAddress, chunkSize);
    }
    else
    {
      destination.WriteToBlock(idx, chunkOverhead, baseAddress, chunkSize);
    }
  }
}

void afmm::HeapReflector::Restore( afm::HeapBlockArena& destinationHeap, MemoryReflector& source )
{
  // transfer back data contained in the reflected heap
  typedef afm::HeapBlockArena::Chunk chunk_t;
  const size_type chunkOverhead = afm::HeapBlockArena::ChunkOverhead;
  const size_type heapOverhead = afm::HeapBlockArena::HeapOverhead;

  int blockCount = source.GetBlockCount();
  for (int idx = 0; idx < blockCount; idx++)
  {
    uint64 addrOffset = chunkOverhead;
    if (idx == 0)
    {
      addrOffset += heapOverhead;
    }
    
    void *dstAddress = destinationHeap.GetChunkBaseAddress(idx);
    uint64 chunkSize = destinationHeap.GetChunkSize(idx);
    source.Restore(dstAddress, idx, addrOffset, chunkSize);
  }
}

void afmm::HeapReflector::InitializeMetadata( MemoryReflector& reflector, const afm::HeapBlockArena& sourceHeap )
{
  // just to shorten type declarations
  typedef afm::HeapBlockArena::Chunk chunk_t;
  typedef afm::HeapBlockArena::Metadata metadata_t;
  // constants to make code more clean
  const size_type heapOverhead = afm::HeapBlockArena::HeapOverhead;
  void *reflectedHeapAddress = reflector.GetBlockStartAddress(0);

  // clone every data member
  metadata_t clonedMetadata;
  int blockCount = reflector.GetBlockCount();
  clonedMetadata.chunkCount_ = blockCount;
  // initialize chunk addresses; first chunk is calculated differently
  clonedMetadata.chunks_[0] = (chunk_t *)((char *)reflectedHeapAddress + heapOverhead);
  for (int idx = 1; idx < blockCount; idx++)
  {
    clonedMetadata.chunks_[idx] = (chunk_t *)reflector.GetBlockStartAddress(idx);
  }
  clonedMetadata.chunkSize_ = sourceHeap.metadata_.chunkSize_;

  // this member is the only one which we don't properly clone, but it should not
  // affect the expected behavior, anyway
  clonedMetadata.currentChunk_ = clonedMetadata.chunks_[0];

  clonedMetadata.firstChunk_ = clonedMetadata.chunks_[0];
  clonedMetadata.freeSpace_ = sourceHeap.metadata_.freeSpace_;
  clonedMetadata.initialSize_ = sourceHeap.metadata_.initialSize_;
  clonedMetadata.lastChunk_ = (chunk_t *)reflector.GetBlockStartAddress(blockCount-1);
  clonedMetadata.nextChunkIndex_ = blockCount;
  clonedMetadata.pimpl_ = clonedMetadata.pimpl_; // must be the same so restoration does not 
                                                 // break implementation

  clonedMetadata.poolId_ = sourceHeap.metadata_.poolId_;
  clonedMetadata.totalSize_ = sourceHeap.metadata_.totalSize_;

  // now, copy memory block to target
  reflector.WriteToBlock(0, &clonedMetadata, sizeof(metadata_t));
}

void afmm::HeapReflector::InitializeChunkMetadata( MemoryReflector& reflector, const afm::HeapBlockArena& sourceHeap )
{
  // just to shorten type declarations
  typedef afm::HeapBlockArena::Chunk chunk_t;

  // constants to make code more clean
  const size_type heapOverhead = afm::HeapBlockArena::HeapOverhead;
  const size_type chunkOverhead = afm::HeapBlockArena::ChunkOverhead;
  void *reflectedHeapAddress = reflector.GetBlockStartAddress(0);

  int chunkCount = reflector.GetBlockCount();
  for (int idx = 0; idx < chunkCount; idx++)
  {
    chunk_t dummyChunk; // we will write data here
    chunk_t *srcChunk = (chunk_t *)((char *)sourceHeap.GetChunkBaseAddress(idx) - chunkOverhead);
    chunk_t *destChunk = nullptr;
    chunk_t *nextChunk = nullptr;
    chunk_t *previousChunk = nullptr;
    if (idx < chunkCount - 1) // if it is not the last one
    {
      nextChunk = (chunk_t *)reflector.GetBlockStartAddress(idx+1);
    }
    if (idx == 0) // first block is calculated differently
    {
      destChunk = (chunk_t *)((char *)reflectedHeapAddress + heapOverhead);
      dummyChunk.Size = sourceHeap.metadata_.initialSize_;
    }
    else
    {
      destChunk = (chunk_t *)reflector.GetBlockStartAddress(idx);
      dummyChunk.Size = sourceHeap.GetChunkSize(idx);
      if (idx == 1)
      {
        previousChunk = (chunk_t *)((char *)reflectedHeapAddress + heapOverhead);
      }
      else
      {
        previousChunk = (chunk_t *)reflector.GetBlockStartAddress(idx-1);
      }
    }
    dummyChunk.StartingAddress = (char *)destChunk + chunkOverhead;
    dummyChunk.AllocationCount = srcChunk->AllocationCount;
    dummyChunk.FreeSpace = srcChunk->FreeSpace;
    dummyChunk.Index = srcChunk->Index;
    dummyChunk.Next = nextChunk;
    
    // calculate chunk used space and use it to set next free address
    uint64 usedSpace = (uint64)(srcChunk->NextFreeAddress - srcChunk->StartingAddress);
    dummyChunk.NextFreeAddress = dummyChunk.StartingAddress + usedSpace;
    dummyChunk.Previous = previousChunk;
    if (idx == 0)
    {
      reflector.WriteToBlock(0, heapOverhead, &dummyChunk, sizeof(chunk_t));
    }
    else
    {
      reflector.WriteToBlock(idx, &dummyChunk, sizeof(chunk_t));
    }
  }
}

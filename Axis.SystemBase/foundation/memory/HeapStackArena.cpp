#include "HeapStackArena.hpp"
#include "RelativePointer.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include <mutex>

namespace afm = axis::foundation::memory;

struct afm::HeapStackArena::Chunk
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

struct afm::HeapStackArena::Pimpl
{
public:
  std::mutex Mutex;
};

namespace {
  typedef struct 
  {
    uint64 Size;
    bool IsFree;
    int ChunkIndex;
  } MemoryBlock;
}

#define ALLOCATION_OVERHEAD     sizeof(MemoryBlock) + sizeof(uint64)
#define ALLOCATION_ALIGNMENT    8ui64

afm::HeapStackArena::HeapStackArena( uint64 initialSize, uint64 chunkSize )
{
  pimpl_ = new Pimpl();
  initialSize_ = initialSize;
  chunkSize_ = chunkSize;
  totalSize_ = 0;
  chunkCount_ = 0;  
  nextChunkIndex_ = 0;
  Init();
}

afm::HeapStackArena::~HeapStackArena( void )
{
  Chunk *chunk = firstChunk_;
  while (chunk != nullptr)
  {
    delete [] chunk->StartingAddress;
    Chunk *aux = chunk;
    chunk = chunk->Next;
    delete aux;
  }
  delete pimpl_;
  firstChunk_ = nullptr;
  lastChunk_ = nullptr;
  chunkCount_ = 0;
  nextChunkIndex_ = 0;
  pimpl_ = nullptr;
}

void afm::HeapStackArena::Init( void )
{
  firstChunk_ = CreateChunk(initialSize_, nullptr);
  lastChunk_ = firstChunk_;
  currentChunk_ = firstChunk_;
  chunks_[0] = firstChunk_;
  nextChunkIndex_ = 1;
}

void afm::HeapStackArena::Destroy( void ) const
{
  delete this;
}

afm::HeapStackArena& afm::HeapStackArena::Create( uint64 initialSize, uint64 chunkSize )
{
  return *new afm::HeapStackArena(initialSize, chunkSize);
}

void * afm::HeapStackArena::Allocate( uint64 size )
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);

  // align block to word boundaries
  uint64 paddedSize = (size + ALLOCATION_ALIGNMENT - 1) & ~(ALLOCATION_ALIGNMENT - 1);

  // first, if block is larger than chunk size, then 
  // create a new and exclusive chunk for it
  if (paddedSize + ALLOCATION_OVERHEAD > chunkSize_)
  {
    Chunk *chunk = CreateChunk(paddedSize + ALLOCATION_OVERHEAD, lastChunk_);
    lastChunk_ = chunk;
    return AllocateBlock(paddedSize, chunk);
  }

  // check if we have space in the current chunk
  if (currentChunk_->FreeSpace >= paddedSize + ALLOCATION_OVERHEAD)
  { // allocate here
    return AllocateBlock(paddedSize, currentChunk_);
  }

  // no, so we check for another chunk
  Chunk *c = firstChunk_;
  while (c != nullptr)
  {
    if (c->FreeSpace >= paddedSize + ALLOCATION_OVERHEAD)
    {
      return AllocateBlock(paddedSize, c);
    }
    c = c->Next;
  }

  // no free chunk -- create a new one
  Chunk *newChunk = CreateChunk(chunkSize_, lastChunk_);
  lastChunk_ = newChunk;
  currentChunk_ = newChunk;
  return AllocateBlock(paddedSize, newChunk);
}

void afm::HeapStackArena::Deallocate( const void * ptr )
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);

  // recover block size and its metadata
  uint64 *blockHeader = reinterpret_cast<uint64 *>(
                              reinterpret_cast<uint64>(ptr) - sizeof(uint64));
  uint64 blockSize = *blockHeader;
  MemoryBlock *blockInfo = reinterpret_cast<MemoryBlock *>(reinterpret_cast<uint64>(ptr) + blockSize);

  // fail if block has already been freed
  if (blockInfo->IsFree)
  {
    throw axis::foundation::InvalidOperationException();
  }

  // update chunk information  
  Chunk *c = chunks_[blockInfo->ChunkIndex];
  c->AllocationCount--;
  
  // update block information
  blockInfo->IsFree = true;

  // recover chunk free space if this is a tail block
  if (IsTailAllocation(ptr, blockSize, c))
  {
    ReclaimChunkFreeSpace(c, ptr);
  }
  
  if (c->AllocationCount == 0)
  { // free chunk only if it composes a tail of free chunks
    bool isFreeTail = true;
    Chunk *chunkNode = c->Next;
    while (chunkNode != nullptr)
    {
      if (chunkNode->AllocationCount != 0)
      {
        isFreeTail = false;
        break;
      }
      chunkNode = chunkNode->Next;
    }

    c->FreeSpace = c->Size;
    c->NextFreeAddress = c->StartingAddress;
    if (c != firstChunk_ && isFreeTail)
    { // purge this chunk and any previous that is also free
      Chunk *chunkNode = c->Previous;
      Chunk *lastChunk = c->Previous;
      while (chunkNode != firstChunk_)
      {
        if (chunkNode->AllocationCount != 0)
        {
          break;
        }
        chunkNode = chunkNode->Previous;
        lastChunk = chunkNode;
      }

      Chunk *chunkToFree = lastChunk->Next;
      while (chunkToFree != nullptr)
      {
        Chunk *c = chunkToFree;
        chunkToFree = chunkToFree->Next;

        totalSize_ -= c->Size;
        chunkCount_--;
        delete [] c->StartingAddress;
        delete c;
      }
      nextChunkIndex_ = lastChunk->Index + 1;
      lastChunk->Next = nullptr;
      lastChunk_ = lastChunk;
      currentChunk_ = lastChunk;
    }
  }
}

void afm::HeapStackArena::Obliterate( void )
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);
  // delete every chunk but first
  Chunk *chunk = firstChunk_->Next;
  while (chunk != nullptr)
  {
    delete [] chunk->StartingAddress;
    Chunk *aux = chunk;
    chunk = chunk->Next;
    delete aux;
  }
  // reset initial chunk
  firstChunk_->AllocationCount = 0;
  firstChunk_->FreeSpace = firstChunk_->Size;
  firstChunk_->Next = nullptr;

  // reset members
  chunkCount_ = 1;
  totalSize_ = firstChunk_->Size;
  lastChunk_ = firstChunk_;
  nextChunkIndex_ = 1;
}

uint64 afm::HeapStackArena::GetMaxContiguousAllocatedFreeSpace( void ) const
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);
  uint64 maxSpace = 0;
  Chunk *chunk = firstChunk_;
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

uint64 afm::HeapStackArena::GetNonContiguousAllocatedFreeSpace( void ) const
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);
  uint64 totalSpace = 0;
  Chunk *chunk = firstChunk_;
  while (chunk != nullptr)
  {
    totalSpace += chunk->FreeSpace;
    chunk = chunk->Next;
  }
  return totalSpace;
}

uint64 afm::HeapStackArena::GetTotalSize( void ) const
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);
  return totalSize_;
}

size_type afm::HeapStackArena::GetFragmentationCount( void ) const
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);
  return chunkCount_;
}

void * afm::HeapStackArena::GetChunkBaseAddress( int chunkIndex ) const
{
//   std::lock_guard<std::mutex> guard(pimpl_->Mutex);
  if (chunkIndex < 0 || chunkIndex >= chunkCount_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return chunks_[chunkIndex]->StartingAddress;
}

afm::HeapStackArena::Chunk * afm::HeapStackArena::CreateChunk(uint64 size, Chunk *previous)
{
  if (chunkCount_ == MAX_BLOCK_ARENA_CHUNK_SIZE)
  { // cannot create any more chunks
    throw axis::foundation::OutOfMemoryException();
  }

  Chunk *c = new Chunk();
  try
  {
    c->StartingAddress = new char[size];
  }
  catch (...)
  {
    delete c;
    throw axis::foundation::OutOfMemoryException();
  }
  c->Index = nextChunkIndex_;
  c->NextFreeAddress = c->StartingAddress;
  c->Size = size;
  c->FreeSpace = size;
  c->AllocationCount = 0;
  c->Next = nullptr;
  c->Previous = previous;
  if (previous != nullptr) previous->Next = c;
  chunkCount_++;
  totalSize_ += size;
  chunks_[nextChunkIndex_] = c;
  nextChunkIndex_++;
  return c;
}

void * afm::HeapStackArena::AllocateBlock( uint64 size, Chunk *chunk )
{
  char *addr = chunk->NextFreeAddress;
  char *nextFreeAddr = reinterpret_cast<char *>(
                       reinterpret_cast<uint64>(addr) + size + ALLOCATION_OVERHEAD);

  uint64 *sizeHeader = reinterpret_cast<uint64 *>(addr);
  char *blockStartAddress = addr + sizeof(uint64);

  // write allocation size
  *sizeHeader = size;

  // init allocation block tail header
  MemoryBlock *block = reinterpret_cast<MemoryBlock *>(blockStartAddress + size);
  block->IsFree = false;
  block->Size = size;
  block->ChunkIndex = chunk->Index;

  // update chunk
  chunk->FreeSpace -= (size + ALLOCATION_OVERHEAD);
  chunk->AllocationCount++;
  chunk->NextFreeAddress = nextFreeAddr;

  return (void *)blockStartAddress;
}

bool afm::HeapStackArena::IsTailAllocation( const void * ptr, uint64 size, const Chunk *chunk ) const
{
  return ((const char *)ptr + size + sizeof(MemoryBlock)) == chunk->NextFreeAddress;
}

void afm::HeapStackArena::ReclaimChunkFreeSpace( Chunk *chunk, const void *tailAllocationPtr )
{
  // recover tail block size and its metadata
  uint64 *blockHeader = 
    reinterpret_cast<uint64 *>(reinterpret_cast<uint64>(tailAllocationPtr) - sizeof(uint64));
  uint64 blockSize = *blockHeader;
  MemoryBlock *blockInfo = 
    reinterpret_cast<MemoryBlock *>(reinterpret_cast<uint64>(tailAllocationPtr) + blockSize);

  // trivial case: the entire chunk is now free
  if (chunk->AllocationCount == 0)
  {
    chunk->FreeSpace = chunk->Size;
    chunk->NextFreeAddress = chunk->StartingAddress;
    return;
  }

  uint64 reclaimedSpace = 0;
  uint64 chunkStartAddress = reinterpret_cast<uint64>(chunk->StartingAddress);
  while (reinterpret_cast<uint64>(blockInfo) > chunkStartAddress)
  {
    if (!blockInfo->IsFree)
    {
      // mark as free space
      chunk->NextFreeAddress = reinterpret_cast<char *>(blockInfo) + sizeof(MemoryBlock);
      chunk->FreeSpace += reclaimedSpace;
      return;
    }
    uint64 totalBlockSize = blockInfo->Size + ALLOCATION_OVERHEAD;
    reclaimedSpace += totalBlockSize;
    uint64 newBlockPtr = reinterpret_cast<uint64>(blockInfo) - totalBlockSize;
    blockInfo = reinterpret_cast<MemoryBlock *>(newBlockPtr);
  }

}

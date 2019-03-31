#include "HeapBlockArena.hpp"
#include "RelativePointer.hpp"
#include "HeapBlockArena_Pimpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace afm = axis::foundation::memory;

const size_type afm::HeapBlockArena::ChunkOverhead = sizeof(Chunk);
const size_type afm::HeapBlockArena::AllocationOverhead = 0;
const size_type afm::HeapBlockArena::HeapOverhead = sizeof(HeapBlockArena);

constexpr auto ALLOCATION_ALIGNMENT = 16ui64;
template<typename T>
constexpr auto PAD_ADDRESS(T x) { return ((uint64)(x) + ALLOCATION_ALIGNMENT - 1ui64) & ~(ALLOCATION_ALIGNMENT - 1ui64); }

afm::HeapBlockArena::HeapBlockArena( uint64 initialSize, uint64 chunkSize, int poolId )
{
  // the only available way to create this object is by the static method Create, which in turn
  // uses the placement-new to create a contiguous memory object. So it is safe to use placement-new
  // here too.
  metadata_.poolId_ = poolId;
  metadata_.pimpl_ = new Pimpl();
  metadata_.initialSize_ = initialSize;
  metadata_.chunkSize_ = chunkSize;
  metadata_.totalSize_ = 0;
  metadata_.chunkCount_ = 0;  
  metadata_.nextChunkIndex_ = 0;
  Init();
}

afm::HeapBlockArena::~HeapBlockArena( void )
{
  Chunk *chunk = metadata_.firstChunk_->Next; // first chunk is the object itself
  while (chunk != nullptr)
  {
    Chunk *aux = chunk;
    chunk = chunk->Next;
    delete aux;
  }
  delete metadata_.pimpl_;
  metadata_.firstChunk_ = nullptr;
  metadata_.lastChunk_ = nullptr;
  metadata_.chunkCount_ = 0;
  metadata_.nextChunkIndex_ = 0;
  metadata_.pimpl_ = nullptr;
}

uint64 afm::HeapBlockArena::GetMaxContiguousAllocatedFreeSpace( void ) const
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
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

uint64 afm::HeapBlockArena::GetNonContiguousAllocatedFreeSpace( void ) const
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
  uint64 totalSpace = 0;
  Chunk *chunk = metadata_.firstChunk_;
  while (chunk != nullptr)
  {
    totalSpace += chunk->FreeSpace;
    chunk = chunk->Next;
  }
  return totalSpace;
}

uint64 afm::HeapBlockArena::GetTotalSize( void ) const
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
  return metadata_.totalSize_;
}

size_type afm::HeapBlockArena::GetFragmentationCount( void ) const
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
  return metadata_.chunkCount_;
}

void * afm::HeapBlockArena::GetChunkBaseAddress( int chunkIndex ) const
{
  // Lock removed because it really slows down overall performance due to
  // code serialization when using relative pointers. It might introduce
  // inconsistent states, though.
  // std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
  if (chunkIndex < 0 || chunkIndex >= metadata_.chunkCount_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return metadata_.chunks_[chunkIndex]->StartingAddress;
}

uint64 afm::HeapBlockArena::GetChunkSize( int chunkIndex ) const
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
  if (chunkIndex < 0 || chunkIndex >= metadata_.chunkCount_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return metadata_.chunks_[chunkIndex]->Size;
}



void afm::HeapBlockArena::Init( void )
{
  /*
  Chunk *c = reinterpret_cast<Chunk *>(reinterpret_cast<char *>(this) + sizeof(HeapBlockArena));
  c->StartingAddress = reinterpret_cast<char *>(c) + ChunkOverhead;
  c->Index = metadata_.nextChunkIndex_;
  c->NextFreeAddress = c->StartingAddress;
  c->Size = metadata_.initialSize_;
  c->FreeSpace = metadata_.initialSize_;
  c->AllocationCount = 0;
  c->Next = nullptr;
  c->Previous = nullptr;
  */
  Chunk *c = reinterpret_cast<Chunk *>(PAD_ADDRESS(reinterpret_cast<char *>(this) + sizeof(HeapBlockArena)));
  c->StartingAddress = reinterpret_cast<char *>(PAD_ADDRESS(reinterpret_cast<char *>(c) + ChunkOverhead));
  c->Index = metadata_.nextChunkIndex_;
  c->NextFreeAddress = c->StartingAddress;
  c->Size = metadata_.initialSize_;
  c->FreeSpace = metadata_.initialSize_;
  c->AllocationCount = 0;
  c->Next = nullptr;
  c->Previous = nullptr;

  metadata_.chunkCount_++;
  metadata_.totalSize_ = metadata_.initialSize_;

  metadata_.firstChunk_ = c;
  metadata_.lastChunk_ = metadata_.firstChunk_;
  metadata_.currentChunk_ = metadata_.firstChunk_;
  metadata_.chunks_[0] = metadata_.firstChunk_;
  metadata_.nextChunkIndex_ = 1;
}

void afm::HeapBlockArena::Destroy( void ) const
{
  delete this;
}

afm::HeapBlockArena& afm::HeapBlockArena::Create( uint64 initialSize, uint64 chunkSize, int poolId )
{
  uint64 objectOverhead = sizeof(HeapBlockArena);
  objectOverhead += PAD_ADDRESS(ChunkOverhead);
  void *memArea = new char[initialSize + objectOverhead + ALLOCATION_ALIGNMENT];
  new (memArea) afm::HeapBlockArena(initialSize, chunkSize, poolId);
  return *reinterpret_cast<HeapBlockArena *>(memArea);
}

afm::RelativePointer afm::HeapBlockArena::Allocate( uint64 size )
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);

  // first, if block is larger than chunk size, then 
  // create a new and exclusive chunk for it
  uint64 paddedSize = PAD_ADDRESS(size);
  if (size > metadata_.chunkSize_)
  {
    Chunk *chunk = CreateChunk(size, metadata_.lastChunk_);
    metadata_.lastChunk_ = chunk;
    return AllocateBlock(size, chunk);
  }

  // check if we have space in the current chunk
  uint64 paddedAddress = PAD_ADDRESS(metadata_.currentChunk_->NextFreeAddress);
  uint64 paddingSpace = paddedAddress - (uint64)metadata_.currentChunk_->NextFreeAddress;
  if (metadata_.currentChunk_->FreeSpace >= size + paddingSpace)
  { // allocate here
    return AllocateBlock(size, metadata_.currentChunk_);
  }

  // no, so we check for another chunk
  Chunk *c = metadata_.firstChunk_;
  while (c != nullptr)
  {
    uint64 paddedAddress = PAD_ADDRESS(c->NextFreeAddress);
    uint64 paddingSpace = paddedAddress - (uint64)c->NextFreeAddress;
    if (c->FreeSpace >= size + paddingSpace)
    {
      return AllocateBlock(size, c);
    }
    c = c->Next;
  }

  // no free chunk -- create a new one
  Chunk *newChunk = CreateChunk(PAD_ADDRESS(metadata_.chunkSize_), metadata_.lastChunk_);
  metadata_.lastChunk_ = newChunk;
  metadata_.currentChunk_ = newChunk;
  return AllocateBlock(size, newChunk);
}

void afm::HeapBlockArena::Deallocate( const RelativePointer& ptr )
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
  if (ptr.poolId_ != metadata_.poolId_ || ptr.chunkId_ >= metadata_.chunkCount_)
  {
    throw axis::foundation::InvalidOperationException();
  }

  // get chunk information  
  Chunk *c = metadata_.chunks_[ptr.chunkId_];
  c->AllocationCount--;
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
    if (c != metadata_.firstChunk_ && isFreeTail)
    { // purge this chunk and any previous that is also free
      Chunk *chunkNode = c->Previous;
      Chunk *lastChunk = c->Previous;
      while (chunkNode != metadata_.firstChunk_)
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

        metadata_.totalSize_ -= c->Size;
        metadata_.chunkCount_--;
        delete c;
      }
      metadata_.nextChunkIndex_ = lastChunk->Index + 1;
      lastChunk->Next = nullptr;
      metadata_.lastChunk_ = lastChunk;
      metadata_.currentChunk_ = lastChunk;
    }
  }
}

void afm::HeapBlockArena::Obliterate( void )
{
  std::lock_guard<std::mutex> guard(metadata_.pimpl_->Mutex);
  // delete every chunk but first
  Chunk *chunk = metadata_.firstChunk_->Next;
  while (chunk != nullptr)
  {
    Chunk *aux = chunk;
    chunk = chunk->Next;
    delete aux;
  }
  // reset initial chunk
  metadata_.firstChunk_->AllocationCount = 0;
  metadata_.firstChunk_->FreeSpace = metadata_.firstChunk_->Size;
  metadata_.firstChunk_->Next = nullptr;

  // reset members
  metadata_.chunkCount_ = 1;
  metadata_.totalSize_ = metadata_.firstChunk_->Size;
  metadata_.lastChunk_ = metadata_.firstChunk_;
  metadata_.nextChunkIndex_ = 1;
}

afm::HeapBlockArena::Chunk * afm::HeapBlockArena::CreateChunk(uint64 size, Chunk *previous)
{
  uint64 totalMemBlockSize = PAD_ADDRESS(size) + PAD_ADDRESS(ChunkOverhead) + ALLOCATION_ALIGNMENT;  
  if (metadata_.chunkCount_ == MAX_BLOCK_ARENA_CHUNK_SIZE)
  { // cannot create any more chunks
    throw axis::foundation::OutOfMemoryException();
  }
  char *memBlock = nullptr;
  try
  {
//    memBlock = new char[size + ChunkOverhead];
    memBlock = new char[totalMemBlockSize];
  }
  catch (...)
  {
  	delete memBlock;
    throw axis::foundation::OutOfMemoryException();
  }
  Chunk *c = reinterpret_cast<Chunk *>(memBlock);
//  c->StartingAddress = memBlock + ChunkOverhead;
  c->StartingAddress = reinterpret_cast<char *>(PAD_ADDRESS(memBlock + ChunkOverhead));
  c->Index = metadata_.nextChunkIndex_;
  c->NextFreeAddress = c->StartingAddress;
  c->Size = size;
  c->FreeSpace = size;
  c->AllocationCount = 0;
  c->Next = nullptr;
  c->Previous = previous;
  if (previous != nullptr) previous->Next = c;
  metadata_.chunkCount_++;
  metadata_.totalSize_ += size;
  metadata_.chunks_[metadata_.nextChunkIndex_] = c;
  metadata_.nextChunkIndex_++;
  return c;
}

afm::RelativePointer afm::HeapBlockArena::AllocateBlock( uint64 size, Chunk *chunk )
{
  // align block address to a pre-defined bit boundary
  uint64 addr = reinterpret_cast<uint64>(chunk->NextFreeAddress);
  uint64 paddedAddr = PAD_ADDRESS(addr); // (addr + ALLOCATION_ALIGNMENT - 1) & ~(ALLOCATION_ALIGNMENT - 1);
  char *blockAddr = reinterpret_cast<char *>(paddedAddr);
  char *nextFreeAddr = blockAddr + PAD_ADDRESS(size);
  chunk->FreeSpace -= (reinterpret_cast<uint64>(nextFreeAddr) - addr); // (size + paddedAddr - addr);
  chunk->AllocationCount++;
  chunk->NextFreeAddress = nextFreeAddr;

  uint64 relativeAddress = paddedAddr - reinterpret_cast<uint64>(chunk->StartingAddress);
  return RelativePointer(relativeAddress, chunk->Index, metadata_.poolId_);
}

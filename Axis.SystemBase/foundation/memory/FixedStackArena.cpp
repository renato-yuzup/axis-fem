#include "FixedStackArena.hpp"
#include "foundation/OutOfMemoryException.hpp"
#include "foundation/ArgumentException.hpp"

namespace afm = axis::foundation::memory;

struct afm::FixedStackArena::MemoryBlock
{
public:
  MemoryBlock *Next;
  MemoryBlock *Previous;
  bool isFree;
  unsigned int DataSize : 24;
};

afm::FixedStackArena::FixedStackArena( uint64 heapSize )
{
  try
  {
    arena_ = new char [heapSize];
    totalSize_ = heapSize;
    Reset();
  }
  catch (...)
  {
  	throw axis::foundation::OutOfMemoryException();
  }
}

afm::FixedStackArena::~FixedStackArena( void )
{
  if (arena_) delete [] arena_;
  arena_ = NULL;
  firstBlock_ = NULL;
  lastBlock_ = NULL;
  blockCount_ = 0;
  freeSpace_ = 0;
  totalSize_ = 0;
}

afm::FixedStackArena& afm::FixedStackArena::Create( uint64 heapSize )
{
  return *new afm::FixedStackArena(heapSize);
}

void afm::FixedStackArena::Destroy( void ) const
{
  delete this;
}

void * afm::FixedStackArena::Allocate( int size )
{
  if (GetFreeSpace() < size)
  {
    throw axis::foundation::OutOfMemoryException();
  }
  uint64 usedSpace = size + sizeof(MemoryBlock);
  MemoryBlock *block = reinterpret_cast<MemoryBlock *>(nextFreeAddress_);
  block->Previous = lastBlock_;
  block->Next = NULL;
  block->isFree = false;
  block->DataSize = size;  
  blockCount_++;
  if (lastBlock_ != NULL)
  {
    lastBlock_->Next = block;
  }
  lastBlock_ = block;
  if (firstBlock_ == NULL)
  {
    firstBlock_ = block;
  }
  nextFreeAddress_ = reinterpret_cast<void *>(
    reinterpret_cast<uint64>(nextFreeAddress_) + usedSpace);
  freeSpace_ -= usedSpace;
  return reinterpret_cast<void *>(
    reinterpret_cast<uint64>(block) + sizeof(MemoryBlock));
}

void afm::FixedStackArena::Deallocate( const void * const ptr )
{
  if (!ContainsAddress(ptr))
  { 
    throw axis::foundation::ArgumentException();
  }

  // locate memory block structure
  MemoryBlock *block = reinterpret_cast<MemoryBlock *>(
    reinterpret_cast<uint64>(ptr) - sizeof(MemoryBlock));
  block->isFree = true;

  // try to clear all free blocks in the tail of the chain
  MemoryBlock *b = lastBlock_;
  uint64 blocksToFree = 0;
  while (b != NULL)
  {
    if (!b->isFree)
    {
      // free everything after this block
      nextFreeAddress_ = reinterpret_cast<void *>(
        reinterpret_cast<uint64>(b) + sizeof(MemoryBlock) + b->DataSize);
      blockCount_ -= blocksToFree;
      uint64 usedSpace = reinterpret_cast<uint64>(nextFreeAddress_) - reinterpret_cast<uint64>(arena_);
      freeSpace_ = totalSize_ - usedSpace;
      lastBlock_ = b;
      break;
    }
    b = b->Previous;
    blocksToFree++;
  }

  if (blockCount_ == blocksToFree && b == NULL)
  { // just reset stack
    Reset();
  }
}

void afm::FixedStackArena::Obliterate( void )
{
  Reset();
}

uint64 afm::FixedStackArena::GetFreeSpace( void ) const
{
  // usable free space
  return freeSpace_ - sizeof(MemoryBlock);
}

uint64 afm::FixedStackArena::GetTotalSize( void ) const
{
  return totalSize_;
}

bool afm::FixedStackArena::ContainsAddress( const void * const ptr ) const
{
  uint64 minAddr = reinterpret_cast<uint64>(arena_);
  uint64 maxAddr = minAddr + totalSize_ - 1;
  uint64 addr = reinterpret_cast<uint64>(ptr);
  return addr <= maxAddr && addr >= minAddr;
}

void afm::FixedStackArena::Reset( void )
{
  firstBlock_ = NULL;
  lastBlock_ = NULL;
  freeSpace_ = totalSize_;
  blockCount_ = 0;
  nextFreeAddress_ = arena_;
}

void * axis::foundation::memory::FixedStackArena::GetBaseAddress( void ) const
{
  return arena_;
}

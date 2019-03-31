#include "UnsyncStackArena.hpp"
#include <assert.h>
#include "arena_helper.hpp"
#include "foundation/ArenaMemoryExhaustedException.hpp"
#include "foundation/ArenaStackMismatchException.hpp"
#include "foundation/ArenaException.hpp"

using axis::foundation::memory::UnsyncStackArena;

const uint64 axis::foundation::memory::UnsyncStackArena::minimalAllocatedSize = 4000000;
const uint64 axis::foundation::memory::UnsyncStackArena::defaultInitialAllocatedSize = 4000000;


axis::foundation::memory::UnsyncStackArena::StackAllocator::StackAllocator( const StackAllocator& allocator )
{
  stackStartingAddress_ = allocator.stackStartingAddress_;
  nextAllocationAddress_ = allocator.nextAllocationAddress_;
  arena_ = allocator.arena_;
  stackLevel_ = allocator.stackLevel_;
}

axis::foundation::memory::UnsyncStackArena::StackAllocator::~StackAllocator( void )
{
  uint64 stackSize = (uint64)nextAllocationAddress_ - (uint64)stackStartingAddress_;
  arena_->PurgeStack(stackStartingAddress_, stackSize, stackLevel_);
  arena_ = NULL;
}

void * axis::foundation::memory::UnsyncStackArena::StackAllocator::Allocate( uint64 chunkSize )
{
  UnsyncStackArena::AllocationData allocData =
        arena_->Allocate(chunkSize, nextAllocationAddress_, stackLevel_);
  nextAllocationAddress_ = (void *)
        ((uint64)allocData.AllocationAddress + allocData.AllocatedSize);
  return allocData.AllocationAddress;
}

axis::foundation::memory::UnsyncStackArena::StackAllocator::StackAllocator( UnsyncStackArena *arena, void *stackStartingAddress, int stackLevel )
{
  arena_ = arena;
  stackStartingAddress_ = stackStartingAddress;
  nextAllocationAddress_ = stackStartingAddress;
  stackLevel_ = stackLevel;
}

/****************************************************************************************
******************************* UnsyncStackArena MEMBERS ********************************
****************************************************************************************/
axis::foundation::memory::UnsyncStackArena::~UnsyncStackArena( void )
{
  delete [] reinterpret_cast<byte *>(pool_);
  pool_ = NULL;
}

axis::foundation::memory::UnsyncStackArena::StackAllocator axis::foundation::memory::UnsyncStackArena::GetAllocator( void )
{
  return StackAllocator(this, nextFreeAddress_, currentStackLevel_);
}

uint64 axis::foundation::memory::UnsyncStackArena::GetStackSize( void ) const
{
  return poolSize_;
}

uint64 axis::foundation::memory::UnsyncStackArena::GetFreeSpace( void ) const
{
  return freeSpace_;
}


/****************************************************************************************
************************************ PRIVATE MEMBERS ************************************
****************************************************************************************/
axis::foundation::memory::UnsyncStackArena::UnsyncStackArena( void )
{
  Init(defaultInitialAllocatedSize);
}

axis::foundation::memory::UnsyncStackArena::UnsyncStackArena(uint64 initialAllocatedSize)
{
  Init(initialAllocatedSize);
}

void axis::foundation::memory::UnsyncStackArena::Init( uint64 sizeToAllocate )
{
  pool_ = new byte[sizeToAllocate];
  nextFreeAddress_ = pool_;
  poolSize_ = sizeToAllocate;
  freeSpace_ = sizeToAllocate;
  currentStackLevel_ = 0;
}

axis::foundation::memory::UnsyncStackArena::AllocationData axis::foundation::memory::UnsyncStackArena::Allocate( uint64 chunkSize, const void * const searchStartAddress, int stackLevel )
{
  // resize chunk according to the platform alignment rules
  uint64 alignmentMask = -machineMemoryAlignmentSize; // we are only interested in the 
                                                      // bitwise operation this represents
  uint64 memSize = (chunkSize + (machineMemoryAlignmentSize - 1)) & alignmentMask;
  
  if (stackLevel != currentStackLevel_ || nextFreeAddress_ != searchStartAddress)
  {
    throw axis::foundation::ArenaStackMismatchException();
  }
  
  if (freeSpace_ < memSize)
  {
    throw axis::foundation::ArenaMemoryExhaustedException();
  }

  AllocationData data;
  data.AllocationAddress = nextFreeAddress_;
  nextFreeAddress_ = (void *)((uint64)nextFreeAddress_ + memSize);
  freeSpace_ -= memSize;
  data.AllocatedSize = memSize;
  return data;
}

void axis::foundation::memory::UnsyncStackArena::PurgeStack( void *stackStartAddress, uint64 stackSize, int stackLevel )
{
  if (currentStackLevel_ != stackLevel)
  {
    throw axis::foundation::ArenaException(_T("Unordered stack memory arena rewind detected."));
  }

  // ok, rewind stack
  currentStackLevel_--;
  assert(currentStackLevel_ == 0 && "Unexpected behavior in stack memory arena.");
  freeSpace_ = poolSize_ - ((uint64)stackStartAddress - (uint64)pool_);
  nextFreeAddress_ = stackStartAddress;
}



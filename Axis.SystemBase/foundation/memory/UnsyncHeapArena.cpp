#include "UnsyncHeapArena.hpp"
#include <assert.h>
#include "arena_helper.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/ArenaMemoryExhaustedException.hpp"

namespace afm = axis::foundation::memory;
using afm::UnsyncHeapArena;

namespace {
  struct SegmentRecord
  {
  public:
    void * PreviousSegment;
    void * NextSegment;
    void * LastFreedBlockAddress;
    void * UserSpaceStartingAddress;
    uint64 TotalSize;
    uint64 UserSpaceSize;
    uint64 FreeSpace;
  };
  struct FreeBlockRecord
  {
  public:
    void * PreviousBlockAddress;
    void * NextBlockAddress;
    uint64 BlockSize;
  };
  struct AllocationRecord
  {
  public:
    uint64 AllocatedSize;
    int ReferenceCount;
  };
}


static const int segmentPtrBufSize = 128;
static const int minimalAllocationSize = 2 * sizeof(FreeBlockRecord);
const uint64 afm::UnsyncHeapArena::defaultInitialAllocatedSize = 4000000; // 4 MB
const uint64 afm::UnsyncHeapArena::minimalAllocatedSize = 4000000;  // 4 MB

axis::foundation::memory::UnsyncHeapArena::~UnsyncHeapArena( void )
{
  SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
  while (segment != NULL)
  {
    byte *segmentAddress = reinterpret_cast<byte *>(segment);
    segment = reinterpret_cast<SegmentRecord *>(segment->PreviousSegment);
    delete [] segmentAddress;
  }
  poolTail_ = NULL;
  poolHead_ = NULL;
  segmentCount_ = 0;
  totalAllocationSize_ = 0;
}

void * axis::foundation::memory::UnsyncHeapArena::Allocate( uint64 chunkSize )
{
  // resize chunk according to the platform alignment rules
  uint64 alignmentMask = -machineMemoryAlignmentSize; // we are only interested in the 
  // bitwise operation this represents
  uint64 memSize = (chunkSize + (machineMemoryAlignmentSize - 1)) & alignmentMask;

  void *address = SearchFreeChunkInArena(memSize);
  if (address == NULL)
  {
    if (!CanGrow())
    {
      throw axis::foundation::ArenaMemoryExhaustedException();
    }
    ExpandArena(memSize);

    // try again
    address = SearchFreeChunkInArena(memSize);
    if (address == NULL)
    {
      // that's it, we cannot advance further...
      throw axis::foundation::ArenaMemoryExhaustedException();
    }
  }
  // allocate block
  SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
  segment->LastFreedBlockAddress = 
    (void *)((uint64)segment->LastFreedBlockAddress + memSize);
  segment->FreeSpace -= memSize;
  return address;
}

void axis::foundation::memory::UnsyncHeapArena::Obliterate( void )
{
  // turn back arena to its initial state
  SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
  while (segment != poolHead_)
  {
    byte *segmentAddress = reinterpret_cast<byte *>(segment);
    segment = reinterpret_cast<SegmentRecord *>(segment->PreviousSegment);
    delete [] segmentAddress;
  }
  segment->NextSegment = NULL;
  segment->LastFreedBlockAddress = segment->UserSpaceStartingAddress;
  segment->FreeSpace = segment->UserSpaceSize;
  poolTail_ = poolHead_;
  segmentCount_ = 1;
  totalAllocationSize_ = segment->FreeSpace;
}

int axis::foundation::memory::UnsyncHeapArena::GetFragmentationCount( void ) const
{
 return segmentCount_;
}

uint64 axis::foundation::memory::UnsyncHeapArena::GetArenaSize( void ) const
{
 return totalAllocationSize_;
}

bool axis::foundation::memory::UnsyncHeapArena::CanGrow( void ) const
{
 return isFixedSize_;
}

uint64 axis::foundation::memory::UnsyncHeapArena::GetUnallocatedSize( void ) const
{
 SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
 return segment->FreeSpace;
}



/****************************************************************************************
************************************ PRIVATE MEMBERS ************************************
****************************************************************************************/
UnsyncHeapArena::UnsyncHeapArena( void )
{
  Init(defaultInitialAllocatedSize, false);
}

UnsyncHeapArena::UnsyncHeapArena(uint64 initialAllocatedSize)
{
  if (initialAllocatedSize == 0)
  {
    throw axis::foundation::ArgumentException(_T("initialAllocatedSize"));
  }
  Init(initialAllocatedSize, false);
}

UnsyncHeapArena::UnsyncHeapArena( bool fixedSize )
{
  Init(defaultInitialAllocatedSize, fixedSize);
}

UnsyncHeapArena::UnsyncHeapArena(uint64 initialAllocatedSize, bool fixedSize)
{
  if (initialAllocatedSize == 0)
  {
    throw axis::foundation::ArgumentException(_T("initialAllocatedSize"));
  }
  Init(initialAllocatedSize, fixedSize);
}

void UnsyncHeapArena::Init( uint64 sizeToAllocate, bool fixedSize )
{
  byte *segment = NULL;

  // should any allocation error occur, let it bubble up
  segment = AllocateSegment(sizeToAllocate);
  WriteSegmentData(segment, sizeToAllocate, NULL, NULL);

  // init member variables
  poolHead_ = segment;
  poolTail_ = segment;
  segmentCount_ = 1;
  isFixedSize_ = fixedSize;
  totalAllocationSize_ = sizeToAllocate;
}

UnsyncHeapArena::byte * axis::foundation::memory::UnsyncHeapArena::AllocateSegment( uint64 size )
{
  if (size == 0)
  {
    throw axis::foundation::ArgumentException(_T("size"));
  }
  // include segment record size
  uint64 segmentSize = size + sizeof(SegmentRecord);
  // resize segment according to the platform alignment rules
  uint64 alignmentMask = -machineMemoryAlignmentSize;
  segmentSize = (segmentSize + (machineMemoryAlignmentSize - 1)) & alignmentMask;
  return new byte[segmentSize];
}

void UnsyncHeapArena::WriteSegmentData( void *segmentStartingAddress, uint64 size, void *previousSegment, void *nextSegment )
{
  int segmentRecordSize = sizeof(SegmentRecord);
  void *userStartingAddr = (void *)((uint64)segmentStartingAddress + segmentRecordSize);

  // user starting must be aligned according to platform restrictions
  uint64 alignmentMask = -machineMemoryAlignmentSize; // we are only interested in the 
                                                      // bitwise operation this represents
  userStartingAddr = (void *)
        (((uint64)userStartingAddr + (machineMemoryAlignmentSize - 1)) & alignmentMask);
  uint64 userSegmentSize = 
        size - ((uint64)userStartingAddr - (uint64)segmentStartingAddress);

  SegmentRecord *record = reinterpret_cast<SegmentRecord *>(segmentStartingAddress);
  record->FreeSpace = userSegmentSize;
  record->UserSpaceSize = userSegmentSize;
  record->NextSegment = nextSegment;
  record->PreviousSegment = previousSegment;
  record->TotalSize = size;
  record->LastFreedBlockAddress = userStartingAddr;
  record->UserSpaceStartingAddress = userStartingAddr;
}

void UnsyncHeapArena::ExpandArena( uint64 minChunkSize )
{
  SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
  uint64 segmentSize = segment->UserSpaceSize;

  // if current segment is empty, but it is too small, retry allocation
  if (segment->FreeSpace == segmentSize && segmentSize < minChunkSize)
  {
    assert(poolTail_ == poolHead_); // this should only happens for the first segment
    void *replacementSegmentAddress = AllocateSegment(minChunkSize);
    WriteSegmentData(replacementSegmentAddress, minChunkSize, NULL, NULL);
    delete [] reinterpret_cast<byte *>(segment);
    poolHead_ = replacementSegmentAddress;
    poolTail_ = replacementSegmentAddress;
    totalAllocationSize_ = minChunkSize;
  }
  else
  {
    if (segmentSize < minChunkSize) segmentSize = minChunkSize;

    void *newSegmentAddress = AllocateSegment(segmentSize);
    SegmentRecord *newSegment = reinterpret_cast<SegmentRecord *>(newSegmentAddress);

    segment->NextSegment = newSegmentAddress;
    WriteSegmentData(newSegmentAddress, segmentSize, segment, NULL);
    poolTail_ = newSegmentAddress;
    segmentCount_++;
    totalAllocationSize_ += minChunkSize;
  }
}

void * UnsyncHeapArena::SearchFreeChunkInArena( uint64 chunkSize )
{
  SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
  if (segment->FreeSpace < chunkSize) return NULL;
  
  return segment->LastFreedBlockAddress;
}


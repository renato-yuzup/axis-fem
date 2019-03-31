#include "SyncedHeapArena.hpp"
#include <assert.h>
#include <boost/thread/mutex.hpp>
#include "arena_helper.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/ArenaMemoryExhaustedException.hpp"

namespace afm = axis::foundation::memory;

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

class afm::SyncedHeapArena::SyncData
{
public:
  boost::mutex mutex;
};


static const int segmentPtrBufSize = 128;
static const int minimalAllocationSize = 2 * sizeof(FreeBlockRecord);
const uint64 afm::SyncedHeapArena::defaultInitialAllocatedSize = 4000000; // 4 MB
const uint64 afm::SyncedHeapArena::minimalAllocatedSize = 4000000;  // 4 MB

afm::SyncedHeapArena::~SyncedHeapArena( void )
{
  delete data_;
}

void * afm::SyncedHeapArena::Allocate( uint64 chunkSize )
{
  data_->mutex.lock();

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

  data_->mutex.unlock();
  return address;
}

void afm::SyncedHeapArena::Obliterate( void )
{
  data_->mutex.lock();

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

  data_->mutex.unlock();
}

int afm::SyncedHeapArena::GetFragmentationCount( void ) const
{
  data_->mutex.lock();
  int count = segmentCount_;
  data_->mutex.unlock();
  return count;
}

uint64 afm::SyncedHeapArena::GetHeapSize( void ) const
{
  data_->mutex.lock();
  uint64 size = totalAllocationSize_;
  data_->mutex.unlock();
  return size;
}

bool afm::SyncedHeapArena::CanGrow( void ) const
{
  return isFixedSize_;
}

uint64 afm::SyncedHeapArena::GetFreeSpace( void ) const
{
  data_->mutex.lock();
  SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
  uint64 freeSpace = segment->FreeSpace;
  data_->mutex.unlock();
  return freeSpace;
}



/****************************************************************************************
************************************ PRIVATE MEMBERS ************************************
****************************************************************************************/
afm::SyncedHeapArena::SyncedHeapArena( void ) : data_(new SyncData())
{
  Init(defaultInitialAllocatedSize, false);
}

afm::SyncedHeapArena::SyncedHeapArena(uint64 initialAllocatedSize) : data_(new SyncData())
{
  if (initialAllocatedSize == 0)
  {
    throw axis::foundation::ArgumentException(_T("initialAllocatedSize"));
  }
  Init(initialAllocatedSize, false);
}

afm::SyncedHeapArena::SyncedHeapArena( bool fixedSize ) : data_(new SyncData())
{
  Init(defaultInitialAllocatedSize, fixedSize);
}

afm::SyncedHeapArena::SyncedHeapArena(uint64 initialAllocatedSize, bool fixedSize) : data_(new SyncData())
{
  if (initialAllocatedSize == 0)
  {
    throw axis::foundation::ArgumentException(_T("initialAllocatedSize"));
  }
  Init(initialAllocatedSize, fixedSize);
}

void afm::SyncedHeapArena::Init( uint64 sizeToAllocate, bool fixedSize )
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

afm::SyncedHeapArena::byte * afm::SyncedHeapArena::AllocateSegment( uint64 size )
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

void afm::SyncedHeapArena::WriteSegmentData( void *segmentStartingAddress, uint64 size, void *previousSegment, void *nextSegment )
{
  int segmentRecordSize = sizeof(SegmentRecord);
  void *userStartingAddr = (void *)((uint64)segmentStartingAddress + segmentRecordSize);

  // user starting must be aligned according to platform restrictions
  uint64 alignmentMask = -machineMemoryAlignmentSize; // we are only interested in the 
  // bitwise operation this represents
  userStartingAddr = (void *)
    (((uint64)userStartingAddr + (machineMemoryAlignmentSize - 1)) & alignmentMask);

  SegmentRecord *record = reinterpret_cast<SegmentRecord *>(segmentStartingAddress);
  record->FreeSpace = size;
  record->UserSpaceSize = size;
  record->NextSegment = nextSegment;
  record->PreviousSegment = previousSegment;
  record->TotalSize = size;
  record->LastFreedBlockAddress = userStartingAddr;
  record->UserSpaceStartingAddress = userStartingAddr;
}

void afm::SyncedHeapArena::ExpandArena( uint64 minChunkSize )
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

void * afm::SyncedHeapArena::SearchFreeChunkInArena( uint64 chunkSize )
{
  SegmentRecord *segment = reinterpret_cast<SegmentRecord *>(poolTail_);
  if (segment->FreeSpace < chunkSize) return NULL;

  return segment->LastFreedBlockAddress;
}


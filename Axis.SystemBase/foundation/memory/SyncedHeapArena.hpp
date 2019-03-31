#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { 

  class System;

  namespace foundation { namespace memory {

    class AXISSYSTEMBASE_API SyncedHeapArena
    {
    public:
      ~SyncedHeapArena(void);
      void *Allocate(uint64 chunkSize);
      void Obliterate(void);

      int GetFragmentationCount(void) const;
      uint64 GetHeapSize(void) const;
      bool CanGrow(void) const;
      uint64 GetFreeSpace(void) const;

      static const uint64 defaultInitialAllocatedSize;
      static const uint64 minimalAllocatedSize;
    private:
      typedef char byte;
      class SyncData;

      SyncedHeapArena(void);
      explicit SyncedHeapArena(uint64 initialAllocatedSize);
      explicit SyncedHeapArena(bool fixedSize);
      SyncedHeapArena(uint64 initialAllocatedSize, bool fixedSize);

      void Init(uint64 sizeToAllocate, bool fixedSize);
      byte *AllocateSegment(uint64 size);
      void WriteSegmentData(void *segmentStartingAddress, uint64 size, void *previousSegment, void *nextSegment);
      void ExpandArena(uint64 minChunkSize);
      void *SearchFreeChunkInArena(uint64 chunkSize);

      SyncData *data_;              // internal synchronization data
      void *poolHead_;              // head segment pool
      void *poolTail_;              // tail segment pool
      int segmentCount_;            // number of allocated segments
      bool isFixedSize_;            // does this pool dynamically grows?
      uint64 totalAllocationSize_;  // total memory allocated

      friend class axis::System;

      SyncedHeapArena(const SyncedHeapArena&);
      SyncedHeapArena& operator =(const SyncedHeapArena&);
    };

  } } } // namespace axis::foundation::memory

#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { 

  class System;

  namespace foundation { namespace memory {

    class AXISSYSTEMBASE_API UnsyncStackArena
    {
    public:
      class StackAllocator
      {
      public:
        StackAllocator(const StackAllocator& allocator);
        ~StackAllocator(void);

        void *Allocate(uint64 chunkSize);
      private:
        StackAllocator(UnsyncStackArena *arena, void *stackStartingAddress, int stackLevel);
        
        int stackLevel_;
        void *stackStartingAddress_;
        void *nextAllocationAddress_;
        UnsyncStackArena *arena_;

        // disallow creating this object in the free store 
        void *operator new (size_t size);
        void *operator new [](size_t size);

        // it is also prohibited to overwrite existing instances
        StackAllocator& operator =(const StackAllocator& allocator);

        friend class UnsyncStackArena;
      };

      ~UnsyncStackArena(void);
      StackAllocator GetAllocator(void);
      uint64 GetStackSize(void) const;
      uint64 GetFreeSpace(void) const;

      static const uint64 defaultInitialAllocatedSize;
      static const uint64 minimalAllocatedSize;
    private:
      struct AllocationData
      {
        void *AllocationAddress;
        uint64 AllocatedSize;
      };
      typedef char byte;

      UnsyncStackArena(void);
      explicit UnsyncStackArena(uint64 initialAllocatedSize);

      void Init(uint64 sizeToAllocate);
      AllocationData Allocate(uint64 chunkSize, const void * const searchStartAddress, int stackLevel);
      void PurgeStack(void *stackStartAddress, uint64 stackSize, int stackLevel);

      void *pool_;
      void *nextFreeAddress_;
      uint64 poolSize_;
      uint64 freeSpace_;
      int currentStackLevel_;

      friend class axis::System;
      friend class StackAllocator;

      UnsyncStackArena(const UnsyncStackArena&);
      UnsyncStackArena& operator =(const UnsyncStackArena&);
    };

  } } } // namespace axis::foundation::memory

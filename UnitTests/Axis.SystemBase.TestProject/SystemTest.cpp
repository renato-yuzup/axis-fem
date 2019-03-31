#include "stdafx.h"
#include <omp.h>
#include <thread>
#include "CppUnitTest.h"
#include "AxisString.hpp"
#include "System.hpp"
#include "foundation/memory/FixedStackArena.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/HeapStackArena.hpp"
#include "foundation/memory/RelativePointer.hpp"

// Additional memory needed for allocation metadata in stack memory
#define STACK_MEMORY_ALLOCATION_OVERHEAD          24
#define HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD     24

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace axis;
namespace afm = axis::foundation::memory;

namespace AxisSystemBaseTestProject
{
void ConcurrentHeapBlockAlloc(afm::RelativePointer *p, int range, afm::HeapBlockArena& arena)
{
  // thread function for concurrent access simulation
  for (int i = 0; i < range; i++)
  {
    p[i] = arena.Allocate(200);
  }
}
void ConcurrentHeapStackAlloc(void **p, int range, afm::HeapStackArena& arena)
{
  // thread function for concurrent access simulation
  for (int i = 0; i < range; i++)
  {
    p[i] = arena.Allocate(200);
  }
}

TEST_CLASS(SystemTest)
{
public:
  struct MockData
  {
    int data1;
    char data2[60];
    unsigned long data3;
  };

  TEST_METHOD_INITIALIZE(InitSystemToDefaultParameters)
  {
    // initialize with default parameters
    System::Initialize();
  }

  TEST_METHOD_CLEANUP(EnsureSystemFinalize)
  {
    System::Finalize();
  }

  TEST_METHOD(TestSystemDefaultParameters)
  {
    // success of this test suite depends on correct default parameters
    System::SystemParameters params;
    Assert::AreEqual(5000000ui64, params.InitialStringMemorySize);
    Assert::AreEqual(2000000ui64, params.StringMemoryChunkSize);
    Assert::AreEqual(10000000ui64, params.WorkStackMemorySize);
    Assert::AreEqual(80000000ui64, params.InitialModelMemorySize);
    Assert::AreEqual(60000000ui64, params.ModelMemoryChunkSize);
    Assert::AreEqual(40000000ui64, params.InitialGlobalMemorySize);
    Assert::AreEqual(40000000ui64, params.GlobalMemoryChunkSize);
    Assert::AreEqual(omp_get_max_threads(), params.MaxThreadCount);
  }

  TEST_METHOD(TestSystemInitialize)
  {
    // check arena correct size
    Assert::AreEqual(5000000ui64, System::StringMemory().GetTotalSize());
    Assert::AreEqual(10000000ui64, System::StackMemory().GetTotalSize());
    Assert::AreEqual(80000000ui64, System::ModelMemory().GetTotalSize());
    Assert::AreEqual(40000000ui64, System::GlobalMemory().GetTotalSize());

    // none of them must have fragmentation nor consumed free space
    Assert::AreEqual(5000000ui64, System::StringMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(5000000ui64, System::StringMemory().GetNonContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, System::StringMemory().GetFragmentationCount());
      
    Assert::AreEqual(80000000ui64, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(80000000ui64, System::ModelMemory().GetNonContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, System::ModelMemory().GetFragmentationCount());

    Assert::AreEqual(40000000ui64, System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(40000000ui64, System::GlobalMemory().GetNonContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, System::GlobalMemory().GetFragmentationCount());

    // stack memory has overhead due to metadata
    Assert::AreEqual(true, System::StackMemory().GetFreeSpace() < System::StackMemory().GetTotalSize());
  }

  TEST_METHOD(TestHeapBlockArenaSmallAllocation)
  {
    afm::HeapBlockArena& arena = System::GlobalMemory();
    uint64 size = arena.GetTotalSize();
    uint64 stringSize = System::StringMemory().GetMaxContiguousAllocatedFreeSpace();
    uint64 stackSize = System::StackMemory().GetFreeSpace();
    uint64 modelSize = System::ModelMemory().GetMaxContiguousAllocatedFreeSpace();

    // allocate 120 bytes consecutively
    afm::RelativePointer ptr = arena.Allocate(120);
    size -= 128;
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
    ptr = arena.Allocate(120);
    size -= 128;
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
    ptr = arena.Allocate(120);
    size -= 128;
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, arena.GetFragmentationCount());

    // other arenas must still be intact
    Assert::AreEqual(stringSize, System::StringMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapBlockArenaLargeAllocation)
  {
    afm::HeapBlockArena& arena = System::GlobalMemory();
    uint64 size = arena.GetTotalSize();
    uint64 strMemorySize = System::StringMemory().GetMaxContiguousAllocatedFreeSpace();
    uint64 stackSize = System::StackMemory().GetFreeSpace();
    uint64 modelSize = System::ModelMemory().GetMaxContiguousAllocatedFreeSpace();

    // this allocation is large enough to trigger a new chunk allocation
    afm::RelativePointer ptr = arena.Allocate(50000000);
    Assert::AreEqual(40000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(90000000ui64, arena.GetTotalSize());
    Assert::AreEqual(2L, arena.GetFragmentationCount());

    // other arenas must still be intact
    Assert::AreEqual(strMemorySize, System::StringMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());

    // if executed a small allocation, it must happen in the free chunk
    ptr = arena.Allocate(20000000);
    Assert::AreEqual(20000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(90000000ui64, arena.GetTotalSize());
    Assert::AreEqual(2L, arena.GetFragmentationCount());

    // again, other arenas should still be intact
    Assert::AreEqual(strMemorySize, System::StringMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());

    // if a large allocation is requested, but smaller than total chunk capacity,
    // a new chunk allocation must be triggered
    ptr = arena.Allocate(30000000);
    Assert::AreEqual(20000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(130000000ui64, arena.GetTotalSize());
    Assert::AreEqual(3L, arena.GetFragmentationCount());

    // again, other arenas should still be intact
    Assert::AreEqual(strMemorySize, System::StringMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapBlockArenaNonOverlapping)
  {
    afm::RelativePointer ptr[500];
    afm::HeapBlockArena& arena = System::GlobalMemory();
    for (int i = 0; i < 500; i++)
    {
      ptr[i] = arena.Allocate(200);
    }

    // ensures that multiple small allocations do not overlap
    for (int i = 0; i < 500; i++)
    {
      afm::RelativePointer p1 = ptr[i];
      for (int j = 0; j < 500; j++)
      {
        if (i != j)
        {
          // check for overlapping (notice for adjacent nodes)
          if (*p1 >= *ptr[j] && (uint64)*p1 < (uint64)*ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
          else if ((uint64)*p1 + 200 > (uint64)*ptr[j] && (uint64)*p1 + 200 <= (uint64)*ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
        }
      }
    }
  }

  TEST_METHOD(TestHeapBlockArenaDeallocation)
  {
    afm::HeapBlockArena& arena = System::GlobalMemory();
    uint64 globalMemCapacity = arena.GetTotalSize();
    uint64 strMemorySize = System::StringMemory().GetMaxContiguousAllocatedFreeSpace();
    uint64 stackSize = System::StackMemory().GetFreeSpace();
    uint64 modelSize = System::ModelMemory().GetMaxContiguousAllocatedFreeSpace();

    // allocate 120 bytes consecutively
    afm::RelativePointer ptr1 = arena.Allocate(120);
    globalMemCapacity -= 128;
    Assert::AreEqual(globalMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    afm::RelativePointer ptr2 = arena.Allocate(120);
    globalMemCapacity -= 128;
    Assert::AreEqual(globalMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    afm::RelativePointer ptr3 = arena.Allocate(120);
    globalMemCapacity -= 128;
    Assert::AreEqual(globalMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, arena.GetFragmentationCount());

    // other arenas must still be intact
    Assert::AreEqual(strMemorySize, System::StringMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());

    arena.Deallocate(ptr1);
    // de-allocating this one does not raise free space because we only reclaim 
    // free space when the entire chunk is free
    Assert::AreEqual(globalMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalMemCapacity, arena.GetNonContiguousAllocatedFreeSpace());

    // if we allocate again, it should not be in the same area
    afm::RelativePointer newPtr = arena.Allocate(120);
    Assert::AreNotEqual(true, ptr1 == newPtr);

    // de-allocating these should free the entire arena
    arena.Deallocate(ptr2);
    arena.Deallocate(ptr3);
    arena.Deallocate(newPtr);
    Assert::AreEqual(globalMemCapacity + 384, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalMemCapacity + 384, arena.GetNonContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapBlockArenaChunkRetraction)
  {
    afm::HeapBlockArena& arena = System::GlobalMemory();
    uint64 size = arena.GetTotalSize();

    // allocate blocks 
    afm::RelativePointer ptr1 = arena.Allocate(10000000);
    afm::RelativePointer ptr2 = arena.Allocate(10000000);
    afm::RelativePointer ptr3 = arena.Allocate(10000000);

    // until here, only one chunk is needed
    Assert::AreEqual(1L, arena.GetFragmentationCount());
    Assert::AreEqual(10000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());

    // allocate further to force creating new chunks
    afm::RelativePointer ptr4 = arena.Allocate(12000000);
    afm::RelativePointer ptr5 = arena.Allocate(42000000);
    Assert::AreEqual(3L, arena.GetFragmentationCount());
    Assert::AreEqual(28000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(38000000ui64, arena.GetNonContiguousAllocatedFreeSpace());
      
    // now, de-allocate and see if these chunks are freed
    arena.Deallocate(ptr4);
    Assert::AreEqual(3L, arena.GetFragmentationCount()); // does not de-allocate because it is not a tail chunk
    Assert::AreEqual(40000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(50000000ui64, arena.GetNonContiguousAllocatedFreeSpace());

    // initial chunk should never be freed
    arena.Deallocate(ptr1);
    arena.Deallocate(ptr2);
    arena.Deallocate(ptr3);
    Assert::AreEqual(3L, arena.GetFragmentationCount());
    Assert::AreEqual(40000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(80000000ui64, arena.GetNonContiguousAllocatedFreeSpace());

    // freeing this one clears the entire arena
    arena.Deallocate(ptr5);
    Assert::AreEqual(1L, arena.GetFragmentationCount());
    Assert::AreEqual(40000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(40000000ui64, arena.GetNonContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapBlockArenaConcurrentAccess)
  {
    afm::RelativePointer ptr[5000];
    afm::HeapBlockArena& arena = System::GlobalMemory();
      
    // simulate concurrent access into the arena
    std::thread t1(ConcurrentHeapBlockAlloc, ptr, 1250, std::ref(arena)), 
                t2(ConcurrentHeapBlockAlloc, ptr + 1250, 1250, std::ref(arena)), 
                t3(ConcurrentHeapBlockAlloc, ptr + 2500, 1250, std::ref(arena)), 
                t4(ConcurrentHeapBlockAlloc, ptr + 3750, 1250, std::ref(arena));

    // wait threads to finish
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ensures that multiple small allocations do not overlap
    for (int i = 0; i < 5000; i++)
    {
      afm::RelativePointer p1 = ptr[i];
      for (int j = 0; j < 5000; j++)
      {
        if (i != j)
        {
          // check for overlapping (notice for adjacent nodes)
          if (*p1 >= *ptr[j] && (uint64)*p1 < (uint64)*ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
          else if ((uint64)*p1 + 200 > (uint64)*ptr[j] && (uint64)*p1 + 200 <= (uint64)*ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
        }
      }
    }
  }

  TEST_METHOD(TestHeapBlockArenaObliterate)
  {
    afm::HeapBlockArena& arena = System::GlobalMemory();
    uint64 size = arena.GetTotalSize();

    // allocate blocks 
    afm::RelativePointer ptr1 = arena.Allocate(1000000);
    afm::RelativePointer ptr2 = arena.Allocate(1000000);
    afm::RelativePointer ptr3 = arena.Allocate(1000000);
    Assert::AreEqual(size - 3000000, arena.GetMaxContiguousAllocatedFreeSpace());

    // wipe out everything
    arena.Obliterate();
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapStackArenaSmallAllocation)
  {
    afm::HeapStackArena& arena = System::StringMemory();
    uint64 size = arena.GetTotalSize();
    uint64 stackSize = System::StackMemory().GetFreeSpace();
    uint64 modelSize = System::ModelMemory().GetMaxContiguousAllocatedFreeSpace();
    uint64 globalSize = System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace();

    // allocate 120 bytes consecutively
    void * ptr = arena.Allocate(120);
    size -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
    ptr = arena.Allocate(120);
    size -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
    ptr = arena.Allocate(120);
    size -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, arena.GetFragmentationCount());

    // other arenas must still be intact
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalSize, System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapStackArenaLargeAllocation)
  {
    afm::HeapStackArena& arena = System::StringMemory();
    uint64 size = arena.GetTotalSize();
    uint64 stackSize = System::StackMemory().GetFreeSpace();
    uint64 modelSize = System::ModelMemory().GetMaxContiguousAllocatedFreeSpace();
    uint64 globalMemorySize = System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace();

    // this allocation is large enough to trigger a new chunk allocation
    void * ptr = arena.Allocate(6000000);
    Assert::AreEqual(2L, arena.GetFragmentationCount());
    Assert::AreEqual(5000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(11000000ui64 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetTotalSize());

    // other arenas must still be intact
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalMemorySize, System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace());

    // if executed a small allocation, it must happen in the free chunk
    ptr = arena.Allocate(1800000);
    Assert::AreEqual(2L, arena.GetFragmentationCount());
    Assert::AreEqual(3200000ui64 - HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(11000000ui64 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetTotalSize());

    // again, other arenas should still be intact
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalMemorySize, System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace());

    // if a large allocation is requested, but smaller than total chunk capacity,
    // a new chunk allocation must be triggered
    ptr = arena.Allocate(4000000);
    Assert::AreEqual(3L, arena.GetFragmentationCount());
    Assert::AreEqual(3200000ui64 - HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(15000000ui64 + 2*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetTotalSize());

    // again, other arenas should still be intact
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalMemorySize, System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapStackArenaNonOverlapping)
  {
    void * ptr[500];
    afm::HeapStackArena& arena = System::StringMemory();
    for (int i = 0; i < 500; i++)
    {
      ptr[i] = arena.Allocate(200);
    }

    // ensures that multiple small allocations do not overlap
    for (int i = 0; i < 500; i++)
    {
      void * p1 = ptr[i];
      for (int j = 0; j < 500; j++)
      {
        if (i != j)
        {
          // check for overlapping (notice for adjacent nodes)
          if (p1 >= ptr[j] && (uint64)p1 < (uint64)ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
          else if ((uint64)p1 + 200 > (uint64)ptr[j] && (uint64)p1 + 200 <= (uint64)ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
        }
      }
    }
  }

  TEST_METHOD(TestHeapStackArenaFullDeallocation)
  {
    afm::HeapStackArena& arena = System::StringMemory();
    uint64 strMemCapacity = arena.GetTotalSize();
    uint64 stackSize = System::StackMemory().GetFreeSpace();
    uint64 modelSize = System::ModelMemory().GetMaxContiguousAllocatedFreeSpace();
    uint64 globalMemorySize = System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace();

    // allocate 120 bytes consecutively
    void * ptr1 = arena.Allocate(120);
    strMemCapacity -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    void * ptr2 = arena.Allocate(120);
    strMemCapacity -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    void * ptr3 = arena.Allocate(120);
    strMemCapacity -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, arena.GetFragmentationCount());

    // other arenas must still be intact
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalMemorySize, System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace());

    arena.Deallocate(ptr1);
    // de-allocating this one does not raise free space because it was 
    // not a tail de-allocation
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(strMemCapacity, arena.GetNonContiguousAllocatedFreeSpace());

    // if we allocate again, it should not be in the same area
    void * newPtr = arena.Allocate(120);
    Assert::AreNotEqual(true, ptr1 == newPtr);

    // de-allocating these should free the entire arena
    arena.Deallocate(ptr2);
    arena.Deallocate(ptr3);
    arena.Deallocate(newPtr);
    Assert::AreEqual(strMemCapacity + 360 + 3*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(strMemCapacity + 360 + 3*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetNonContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapStackArenaPartialDeallocation)
  {
    afm::HeapStackArena& arena = System::StringMemory();
    uint64 strMemCapacity = arena.GetTotalSize();
    uint64 stackSize = System::StackMemory().GetFreeSpace();
    uint64 modelSize = System::ModelMemory().GetMaxContiguousAllocatedFreeSpace();
    uint64 globalMemorySize = System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace();

    // allocate 120 bytes consecutively
    void * ptr1 = arena.Allocate(120);
    strMemCapacity -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    void * ptr2 = arena.Allocate(120);
    strMemCapacity -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    void * ptr3 = arena.Allocate(120);
    strMemCapacity -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(1L, arena.GetFragmentationCount());

    // other arenas must still be intact
    Assert::AreEqual(stackSize, System::StackMemory().GetFreeSpace());
    Assert::AreEqual(modelSize, System::ModelMemory().GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(globalMemorySize, System::GlobalMemory().GetMaxContiguousAllocatedFreeSpace());

    arena.Deallocate(ptr2);
    // de-allocating this one does not raise free space because it was 
    // not a tail de-allocation
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(strMemCapacity, arena.GetNonContiguousAllocatedFreeSpace());

    // if we allocate again, it should not be in the same area
    void * newPtr = arena.Allocate(120);
    strMemCapacity -= (120 + HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreNotEqual(true, ptr1 == newPtr);

    // still, this one does not trigger space reclaim
    arena.Deallocate(ptr3);
    Assert::AreEqual(strMemCapacity, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(strMemCapacity, arena.GetNonContiguousAllocatedFreeSpace());

    // de-allocate tail block to trigger space reclaim
    arena.Deallocate(newPtr);
    Assert::AreEqual(strMemCapacity + 360 + 3*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(strMemCapacity + 360 + 3*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetNonContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapStackArenaChunkRetraction)
  {
    afm::HeapStackArena& arena = System::StringMemory();
    uint64 size = arena.GetTotalSize();

    // allocate blocks 
    void * ptr1 = arena.Allocate(1200000);
    void * ptr2 = arena.Allocate(1200000);
    void * ptr3 = arena.Allocate(1200000);

    // until here, only one chunk is needed
    Assert::AreEqual(1L, arena.GetFragmentationCount());
    Assert::AreEqual(1400000ui64 - 3*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetMaxContiguousAllocatedFreeSpace());

    // allocate further to force creating new chunks
    void * ptr4 = arena.Allocate(1800000);
    void * ptr5 = arena.Allocate(1800000);
    Assert::AreEqual(3L, arena.GetFragmentationCount());
    Assert::AreEqual(1400000ui64-3*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(1800000ui64-5*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetNonContiguousAllocatedFreeSpace());

    // now, de-allocate and see if these chunks are freed
    arena.Deallocate(ptr4);
    Assert::AreEqual(3L, arena.GetFragmentationCount()); // does not de-allocate because it is not a tail chunk
    Assert::AreEqual(2000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(3600000ui64-4*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetNonContiguousAllocatedFreeSpace());

    // initial chunk should never be freed
    arena.Deallocate(ptr1);
    arena.Deallocate(ptr2);
    arena.Deallocate(ptr3);
    Assert::AreEqual(3L, arena.GetFragmentationCount());
    Assert::AreEqual(5000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(7200000ui64-HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetNonContiguousAllocatedFreeSpace());

    // freeing this one clears the entire arena
    arena.Deallocate(ptr5);
    Assert::AreEqual(1L, arena.GetFragmentationCount());
    Assert::AreEqual(5000000ui64, arena.GetMaxContiguousAllocatedFreeSpace());
    Assert::AreEqual(5000000ui64, arena.GetNonContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestHeapStackArenaConcurrentAccess)
  {
    void * ptr[5000];
    afm::HeapStackArena& arena = System::StringMemory();

    // simulate concurrent access into the arena
    std::thread t1(ConcurrentHeapStackAlloc, ptr, 1250, std::ref(arena)), 
                t2(ConcurrentHeapStackAlloc, ptr + 1250, 1250, std::ref(arena)), 
                t3(ConcurrentHeapStackAlloc, ptr + 2500, 1250, std::ref(arena)), 
                t4(ConcurrentHeapStackAlloc, ptr + 3750, 1250, std::ref(arena));

    // wait threads to finish
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ensures that multiple small allocations do not overlap
    for (int i = 0; i < 5000; i++)
    {
      void * p1 = ptr[i];
      for (int j = 0; j < 5000; j++)
      {
        if (i != j)
        {
          // check for overlapping (notice for adjacent nodes)
          if (p1 >= ptr[j] && (uint64)p1 < (uint64)ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
          else if ((uint64)p1 + 200 > (uint64)ptr[j] && (uint64)p1 + 200 <= (uint64)ptr[j]+200)
          {
            System::Finalize();
            Assert::Fail(_T("Pointer overlapping detected!"));
          }
        }
      }
    }
  }

  TEST_METHOD(TestHeapStackArenaObliterate)
  {
    afm::HeapStackArena& arena = System::StringMemory();
    uint64 size = arena.GetTotalSize();

    // allocate blocks 
    void * ptr1 = arena.Allocate(1000000);
    void * ptr2 = arena.Allocate(1000000);
    void * ptr3 = arena.Allocate(1000000);
    Assert::AreEqual(size - 3000000 - 3*HEAP_STACK_MEMORY_ALLOCATION_OVERHEAD, arena.GetMaxContiguousAllocatedFreeSpace());

    // wipe out everything
    arena.Obliterate();
    Assert::AreEqual(size, arena.GetMaxContiguousAllocatedFreeSpace());
  }

  TEST_METHOD(TestRelativePointer)
  {
    // allocate blocks 
    afm::RelativePointer ptr1 = System::GlobalMemory().Allocate(18000000);
    afm::RelativePointer ptr2 = System::GlobalMemory().Allocate(18000000);
    afm::RelativePointer ptr3 = System::GlobalMemory().Allocate(18000000);
    afm::RelativePointer ptr4 = System::ModelMemory() .Allocate(1000000);
    afm::RelativePointer ptr5 = System::ModelMemory() .Allocate(1000000);
    afm::RelativePointer ptr6 = System::GlobalMemory().Allocate(1000000);

    // de-reference relative pointers
    char *c1 = (char *)*ptr1;
    char *c2 = (char *)*ptr2;
    char *c3 = (char *)*ptr3;
    char *c4 = (char *)*ptr4;
    char *c5 = (char *)*ptr5;
    char *c6 = (char *)*ptr6;
      
    // let's find out if relative pointer mechanism is correct by
    // calculating absolute address in another way and comparing
    char *globalBaseAddress0 = (char *)System::GlobalMemory().GetChunkBaseAddress(0);
    char *globalBaseAddress1 = (char *)System::GlobalMemory().GetChunkBaseAddress(1);
    Assert::AreEqual((uint64)globalBaseAddress0, (uint64)c1); // typecast, otherwise Assert might think that these are strings
    Assert::AreEqual((uint64)globalBaseAddress0 + 18000000, (uint64)c2);
    Assert::AreEqual((uint64)globalBaseAddress1, (uint64)c3);

    char *modelBaseAddress = (char *)System::ModelMemory().GetChunkBaseAddress(0);
    Assert::AreEqual((uint64)modelBaseAddress, (uint64)c4);
    Assert::AreEqual((uint64)modelBaseAddress + 1000000, (uint64)c5);
    Assert::AreEqual((uint64)globalBaseAddress1 + 18000000, (uint64)c6);

    // if everything goes as expected, we can write in the entire range of these 
    // allocated blocks
    for (int i = 0; i < 18000000; i++)
    {
      c1[i] = i % 256;
      c2[i] = i % 256;
      c3[i] = i % 256;
    }
    for (int i = 0; i < 1000000; i++)
    {
      c4[i] = i % 256;
      c5[i] = i % 256;
      c6[i] = i % 256;
    }
  }

  TEST_METHOD(TestStackMemorySimpleAllocation)
  {
    uint64 freeSpace = System::StackMemory().GetFreeSpace();

    // check correct allocation
    void *ptr = System::StackMemory().Allocate(200);
    Assert::AreEqual(freeSpace - 200 - STACK_MEMORY_ALLOCATION_OVERHEAD, System::StackMemory().GetFreeSpace());

    // allocation address must be at the beginning of the arena
    Assert::AreEqual((void *)((uint64)System::StackMemory().GetBaseAddress() + STACK_MEMORY_ALLOCATION_OVERHEAD), ptr);
  }

  TEST_METHOD(TestStackMemoryMultipleAllocation)
  {
    uint64 freeSpace = System::StackMemory().GetFreeSpace();

    // check correct allocation
    void *ptr[500];
    for (int i = 0; i < 500; i++)
    {
      ptr[i] = System::StackMemory().Allocate(200);
      freeSpace -= (200 + STACK_MEMORY_ALLOCATION_OVERHEAD);
      Assert::AreEqual(freeSpace, System::StackMemory().GetFreeSpace());
    }

    // check allocation addresses
    for (int i = 0; i < 500; i++)
    {
      void *expectedAddress = (void *)(
                                (uint64)System::StackMemory().GetBaseAddress() + 
                                (i+1)*STACK_MEMORY_ALLOCATION_OVERHEAD +
                                i*200);
      Assert::AreEqual(expectedAddress, ptr[i]);
    }
  }

  TEST_METHOD(TestStackMemoryDeallocation)
  {
    uint64 usableSpace = System::StackMemory().GetFreeSpace();
    void *ptr1 = System::StackMemory().Allocate(200);
    void *ptr2 = System::StackMemory().Allocate(200);
    void *ptr3 = System::StackMemory().Allocate(200);
    void *ptr4 = System::StackMemory().Allocate(200);
    uint64 freeSpace = System::StackMemory().GetFreeSpace();

    System::StackMemory().Deallocate(ptr3);

    // because de-allocated memory is not a tail block, no free space
    // should have been added
    Assert::AreEqual(freeSpace, System::StackMemory().GetFreeSpace());

    // now, de-allocating a tail block, both spaces should be freed
    System::StackMemory().Deallocate(ptr4);
    freeSpace += 2*(200 + STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(freeSpace, System::StackMemory().GetFreeSpace());

    System::StackMemory().Deallocate(ptr2);
    System::StackMemory().Deallocate(ptr1);
    Assert::AreEqual(usableSpace, System::StackMemory().GetFreeSpace());
  }

  TEST_METHOD(TestStackMemorySimulatedRun)
  {
    uint64 usableSpace = System::StackMemory().GetFreeSpace();
    void *ptr1 = System::StackMemory().Allocate(200);
    void *ptr2 = System::StackMemory().Allocate(200);
    void *ptr3 = System::StackMemory().Allocate(200);
    void *ptr4 = System::StackMemory().Allocate(200);
    uint64 freeSpace = System::StackMemory().GetFreeSpace();

    System::StackMemory().Deallocate(ptr3);

    // because de-allocated memory is not a tail block, no free space
    // should have been added
    Assert::AreEqual(freeSpace, System::StackMemory().GetFreeSpace());

    // now, de-allocating a tail block, both spaces should be freed
    System::StackMemory().Deallocate(ptr4);
    freeSpace += 2*(200 + STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(freeSpace, System::StackMemory().GetFreeSpace());

    void *ptr5 = System::StackMemory().Allocate(200);
    Assert::AreEqual(ptr3, ptr5);
    freeSpace -= (200 + STACK_MEMORY_ALLOCATION_OVERHEAD);
    Assert::AreEqual(freeSpace, System::StackMemory().GetFreeSpace());

    System::StackMemory().Deallocate(ptr2);
    System::StackMemory().Deallocate(ptr1);
    System::StackMemory().Deallocate(ptr5);
    Assert::AreEqual(usableSpace, System::StackMemory().GetFreeSpace());
  }

  TEST_METHOD(TestStackMemoryObliterate)
  {
    uint64 usableSpace = System::StackMemory().GetFreeSpace();
    void *ptr1 = System::StackMemory().Allocate(200);
    void *ptr2 = System::StackMemory().Allocate(200);
    void *ptr3 = System::StackMemory().Allocate(200);
    uint64 freeSpace = System::StackMemory().GetFreeSpace();

    Assert::AreEqual(usableSpace - 3*(200+STACK_MEMORY_ALLOCATION_OVERHEAD), freeSpace);
    System::StackMemory().Obliterate();

    Assert::AreEqual(usableSpace, System::StackMemory().GetFreeSpace());
  }
};
}

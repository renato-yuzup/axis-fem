#pragma once
#include "foundation/mirroring/MemoryMirror.hpp"
#include "foundation/fwd/memory_fwd.hpp"
#include "foundation/fwd/mirroring_fwd.hpp"
#include "foundation/computing/GPUDevice.hpp"

namespace axis { namespace foundation { namespace mirroring { namespace gpu {

/**
 * Provides mirroring service for an entire heap block memory arena.
 */
class GPUHeapMirror : public axis::foundation::mirroring::MemoryMirror
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] memoryHeap   The heap block memory arena to mirror.
   * @param [in,out] targetDevice Target device where mirroring will occur.
   */
  GPUHeapMirror(axis::foundation::memory::HeapBlockArena& memoryHeap, 
                axis::foundation::computing::GPUDevice& targetDevice);
  virtual ~GPUHeapMirror(void);
  virtual void *GetHostBaseAddress(void) const;
  virtual void *GetGPUBaseAddress(void) const;

  virtual void Allocate( void );
  virtual void Deallocate(void);
  virtual void Mirror( void );
  virtual void Restore( void );
private:
  bool allocated_;
  axis::foundation::memory::HeapBlockArena& memoryHeap_;
  axis::foundation::computing::GPUDevice& targetDevice_;
  axis::foundation::mirroring::MemoryReflector *reflector_;
};

} } } } // namespace axis::services::memory::gpu

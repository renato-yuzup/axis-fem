#pragma once
#include "foundation/mirroring/MemoryMirror.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/computing/GPUDevice.hpp"

namespace axis { namespace foundation { namespace mirroring { namespace gpu {

/**
 * Provides mirroring service for a single memory block.
 */
class GPUBlockMirror : public axis::foundation::mirroring::MemoryMirror
{
public:

  /**
   * Constructor.
   *
   * @param blockSize             Size of the memory block.
   * @param [in,out] targetDevice Target GPU device where data will be mirrored.
   */
  GPUBlockMirror(uint64 blockSize, 
    axis::foundation::computing::GPUDevice& targetDevice);
  ~GPUBlockMirror(void);
  virtual void *GetHostBaseAddress(void) const;
  virtual void *GetGPUBaseAddress(void) const;

  /**
   * Returns the block size.
   *
   * @return The block size.
   */
  uint64 GetBlockSize(void) const;

  virtual void Allocate( void );
  virtual void Deallocate(void);
  virtual void Mirror( void );
  virtual void Restore( void );
private:
  bool allocated_;
  uint64 blockSize_;
  void *hostBlockAddress_;
  void *deviceBlockAddress_;
  axis::foundation::computing::GPUDevice& targetDevice_;
};

} } } } // namespace axis::services::memory::gpu

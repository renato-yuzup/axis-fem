#pragma once
#include "foundation/mirroring/MemoryReflector.hpp"
#include "foundation/computing/GPUDevice.hpp"
#include "nocopy.hpp"

namespace axis { namespace foundation { namespace mirroring { namespace gpu {

/**
 * Implements a memory mirroring service between host memory and GPU device.
 */
class GPUHeapReflector : public axis::foundation::mirroring::MemoryReflector
{
public:
  GPUHeapReflector(axis::foundation::computing::GPUDevice& device);
  ~GPUHeapReflector(void);

  virtual void WriteToBlock( int destBlockIdx, uint64 addressOffset, 
    const void *sourceAddress, uint64 dataSize );
  virtual void Restore( void *destinationAddress, int srcBlockIdx, 
    uint64 addressOffset, uint64 dataSize );
private:
  axis::foundation::computing::GPUDevice& device_;

  DISALLOW_COPY_AND_ASSIGN(GPUHeapReflector);
};

} } } } // namespace axis::foundation::mirroring::gpu

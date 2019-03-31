#pragma once
#include <vector>
#include "foundation/mirroring/MemoryAllocator.hpp"
#include "foundation/computing/GPUDevice.hpp"
#include "nocopy.hpp"

namespace axis { namespace foundation { namespace mirroring { namespace gpu {

/**
 * Implements a memory allocator acting on GPU memory.
 */
class GPUMemoryAllocator : public axis::foundation::mirroring::MemoryAllocator
{
public:
  GPUMemoryAllocator(axis::foundation::computing::GPUDevice& device);
  ~GPUMemoryAllocator(void);
  virtual void Allocate( uint64 blockSize );
  virtual axis::foundation::mirroring::MemoryReflector& BuildReflector( void );
private:
  typedef std::pair<void *, uint64> block_descriptor;
  typedef std::vector<block_descriptor> block_list;

  axis::foundation::computing::GPUDevice& device_;
  block_list blocks_;

  DISALLOW_COPY_AND_ASSIGN(GPUMemoryAllocator);
};

} } } } // namespace axis::foundation::mirroring::gpu

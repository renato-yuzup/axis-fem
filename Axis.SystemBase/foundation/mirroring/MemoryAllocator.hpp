#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/fwd/mirroring_fwd.hpp"
#include "nocopy.hpp"

namespace axis { namespace foundation { namespace mirroring {

class AXISSYSTEMBASE_API MemoryAllocator
{
public:
  MemoryAllocator(void);
  virtual ~MemoryAllocator(void);

  virtual void Allocate(uint64 blockSize) = 0;

  virtual MemoryReflector& BuildReflector(void) = 0;
private:
  DISALLOW_COPY_AND_ASSIGN(MemoryAllocator);
};

} } } // namespace axis::foundation::mirroring

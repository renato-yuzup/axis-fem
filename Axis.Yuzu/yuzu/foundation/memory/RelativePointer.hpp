#if !defined(__RELATIVEPOINTER_H__)
#define __RELATIVEPOINTER_H__
#include "yuzu/common/gpu.hpp"

namespace axis {namespace yuzu { namespace foundation { namespace memory {

class HeapBlockArena;

class RelativePointer
{
public:
  GPU_READY RelativePointer(void);
  GPU_READY RelativePointer(const RelativePointer& other);
  GPU_READY ~RelativePointer(void);
  GPU_READY RelativePointer& operator =(const RelativePointer& other);
  GPU_ONLY void *operator *(void);
  GPU_ONLY const void *operator *(void) const;
  GPU_ONLY bool operator ==(const RelativePointer& ptr) const;
  GPU_ONLY bool operator !=(const RelativePointer& ptr) const;
  GPU_ONLY bool IsNull(void) const;
private:
  size_t relativeAddress_;
  int chunkId_;
  int poolId_;

  friend class HeapBlockArena;
};

extern CPU_ONLY void SetGPUArena(void *address);

} } } } // namespace axis::yuzu::foundation::memory

#endif

#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "nocopy.hpp"

namespace axis { namespace foundation { namespace mirroring {

class AXISSYSTEMBASE_API MemoryReflector
{
public:
  MemoryReflector(void);
  virtual ~MemoryReflector(void);

  void AddBlock(void *blockStartingAddr, uint64 blockSize);

  int GetBlockCount(void) const;
  void *GetBlockStartAddress(int blockIndex) const;
  uint64 GetBlockSize(int blockIndex) const;
  
  virtual void WriteToBlock(int destBlockIdx, uint64 addressOffset, const void *sourceAddress, uint64 dataSize) = 0;
  void WriteToBlock(int destBlockIdx, const void *sourceAddress, uint64 dataSize);

  virtual void Restore(void *destinationAddress, int srcBlockIdx, uint64 addressOffset, uint64 dataSize) = 0;
  void Restore(void *destinationAddress, int srcBlockIdx, uint64 dataSize);

private:
  class Pimpl;
  Pimpl *pimpl_;

  DISALLOW_COPY_AND_ASSIGN(MemoryReflector);
};

} } } // namespace axis::foundation::mirroring

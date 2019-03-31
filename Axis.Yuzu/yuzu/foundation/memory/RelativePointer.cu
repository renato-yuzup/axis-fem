#include "RelativePointer.hpp"
#include "yuzu/foundation/memory/HeapBlockArena.hpp"

#define UNDEFINED_ARENA_POOL_INDEX    -1
#define STRING_ARENA_POOL_INDEX       0
#define MODEL_ARENA_POOL_INDEX        1
#define GLOBAL_ARENA_POOL_INDEX       2

namespace ayfm = axis::yuzu::foundation::memory;

__device__ void * axisYuzuArenaAddress;

__global__ void SetGPUArenaKernel(void *address)
{
  axisYuzuArenaAddress = address;
}

GPU_READY ayfm::RelativePointer::RelativePointer( void ) :
  relativeAddress_(0), chunkId_(0), poolId_(-1)
{
  // nothing to do here
}

GPU_READY ayfm::RelativePointer::RelativePointer( const RelativePointer& other )
{
  relativeAddress_ = other.relativeAddress_;
  chunkId_ = other.chunkId_;
  poolId_ = other.poolId_;
}

GPU_READY ayfm::RelativePointer::~RelativePointer( void )
{
  // nothing to do here
}

GPU_READY ayfm::RelativePointer& ayfm::RelativePointer::operator =( const RelativePointer& other )
{
  relativeAddress_ = other.relativeAddress_;
  chunkId_ = other.chunkId_;
  poolId_ = other.poolId_;
  return *this;
}

GPU_ONLY void * ayfm::RelativePointer::operator *( void )
{
  ayfm::HeapBlockArena& arena = *reinterpret_cast<ayfm::HeapBlockArena *>(axisYuzuArenaAddress);
  return reinterpret_cast<void *>(
    reinterpret_cast<uint64>(arena.GetChunkBaseAddress(chunkId_)) + relativeAddress_);
}

GPU_ONLY const void * ayfm::RelativePointer::operator *( void ) const
{
  ayfm::HeapBlockArena& arena = *reinterpret_cast<ayfm::HeapBlockArena *>(axisYuzuArenaAddress);
  return reinterpret_cast<void *>(
    reinterpret_cast<uint64>(arena.GetChunkBaseAddress(chunkId_)) + relativeAddress_);
}

GPU_ONLY bool ayfm::RelativePointer::operator==( const RelativePointer& ptr ) const
{
  return ptr.chunkId_ == chunkId_ && ptr.poolId_ == poolId_ && ptr.relativeAddress_ == relativeAddress_;
}

GPU_ONLY bool ayfm::RelativePointer::operator!=( const RelativePointer& ptr ) const
{
  return !(*this == ptr);
}

CPU_ONLY void ayfm::SetGPUArena(void *address)
{
  SetGPUArenaKernel<<<1,1,0>>>(address);
}

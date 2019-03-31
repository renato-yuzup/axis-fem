#include "GPUMemoryAllocator.hpp"
#include "GPUHeapReflector.hpp"

namespace afc  = axis::foundation::computing;
namespace afmg = axis::foundation::mirroring::gpu;
namespace afmm = axis::foundation::mirroring;

afmg::GPUMemoryAllocator::GPUMemoryAllocator( afc::GPUDevice& device ) :
  device_(device)
{
  // nothing to do here
}

afmg::GPUMemoryAllocator::~GPUMemoryAllocator(void)
{
  // nothing to do here
}

void afmg::GPUMemoryAllocator::Allocate( uint64 blockSize )
{
  void *block = device_.AllocateMemory(blockSize);
  blocks_.push_back(std::make_pair(block, blockSize));
}

afmm::MemoryReflector& afmg::GPUMemoryAllocator::BuildReflector( void )
{
  GPUHeapReflector &reflector = *new GPUHeapReflector(device_);

  int blockCount = (int)blocks_.size();
  for (int i = 0; i < blockCount; i++)
  {
    block_descriptor block = blocks_[i];
    reflector.AddBlock(block.first, block.second);
  }
  blocks_.clear();
  return reflector;
}

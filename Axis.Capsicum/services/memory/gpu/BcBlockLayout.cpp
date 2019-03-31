#include "BcBlockLayout.hpp"

namespace asmo = axis::services::memory;
namespace asmg = axis::services::memory::gpu;


#define MEMORY_WORD_LENGTH  16
#define ALIGN_BLOCK(x) \
  ((x + MEMORY_WORD_LENGTH - 1) & ~(MEMORY_WORD_LENGTH - 1))

static const int baseBlockSize = ALIGN_BLOCK(sizeof(real) + sizeof(uint64));

asmg::BcBlockLayout::BcBlockLayout( int specificDataSize )
{
  blockSize_ = ALIGN_BLOCK(specificDataSize);
}

asmg::BcBlockLayout::~BcBlockLayout( void )
{
  // nothing to do here
}

asmo::MemoryLayout& asmg::BcBlockLayout::Clone( void ) const
{
  return *new BcBlockLayout(blockSize_);
}

size_type asmg::BcBlockLayout::DoGetSegmentSize( void ) const
{
  return blockSize_ + baseBlockSize;
}

void asmg::BcBlockLayout::InitMemoryBlock( void *targetBlock, uint64 dofId )
{
  *((uint64 *)targetBlock) = dofId;
  *(real *)((uint64)targetBlock + sizeof(uint64)) = 0;
}

void * asmg::BcBlockLayout::GetCustomDataAddress( void *bcBaseAddress ) const
{
  return (void *)((uint64)bcBaseAddress + baseBlockSize);
}

real * asmg::BcBlockLayout::GetOutputBucketAddress( void *bcBaseAddress ) const
{
  return (real *)((uint64)bcBaseAddress + sizeof(uint64));
}

uint64 asmg::BcBlockLayout::GetDofId( void *bcBaseAddress ) const
{
  return *(uint64 *)bcBaseAddress;
}

const void * asmg::BcBlockLayout::GetCustomDataAddress( 
  const void *bcBaseAddress ) const
{
  return (void *)((uint64)bcBaseAddress + baseBlockSize);
}

const real * asmg::BcBlockLayout::GetOutputBucketAddress( 
  const void *bcBaseAddress ) const
{
  return (real *)((uint64)bcBaseAddress + sizeof(uint64));
}

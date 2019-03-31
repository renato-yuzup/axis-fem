#include "CBlockLayout.hpp"

#define MEMORY_WORD_LENGTH 16
#define ALIGN_ADDRESS(x)   ((x) + MEMORY_WORD_LENGTH - 1) & ~(MEMORY_WORD_LENGTH - 1)

namespace asmo = axis::services::memory;
namespace asmg = axis::services::memory::gpu;

static const int baseBlockSize = ALIGN_ADDRESS(sizeof(real));

asmg::CBlockLayout::CBlockLayout( int specificDataSize )
{
  blockSize_ = ALIGN_ADDRESS(specificDataSize);
}

asmg::CBlockLayout::~CBlockLayout( void )
{
  // nothing to do here
}

asmo::MemoryLayout& asmg::CBlockLayout::Clone( void ) const
{
  return *new CBlockLayout(blockSize_);
}

size_type asmg::CBlockLayout::DoGetSegmentSize( void ) const
{
  return blockSize_ + baseBlockSize;
}

void asmg::CBlockLayout::InitMemoryBlock( void *targetBlock )
{
  *((real *)targetBlock) = 0;
}

void * asmg::CBlockLayout::GetCustomDataAddress( void *curveBaseAddress ) const
{
  return (void *)((uint64)curveBaseAddress + baseBlockSize);
}

real * asmg::CBlockLayout::GetOutputBucketAddress(void *curveBaseAddress) const
{
  return (real *)curveBaseAddress;
}

const void * asmg::CBlockLayout::GetCustomDataAddress( 
  const void *curveBaseAddress ) const
{
  return (void *)((uint64)curveBaseAddress + baseBlockSize);
}

const real * asmg::CBlockLayout::GetOutputBucketAddress( 
  const void *curveBaseAddress ) const
{
  return (real *)curveBaseAddress;
}

#include "NoMemoryLayout.hpp"

namespace asmo = axis::services::memory;

asmo::NoMemoryLayout::NoMemoryLayout( size_type blockSize )
{
  blockSize_ = blockSize;
}

asmo::NoMemoryLayout::~NoMemoryLayout( void )
{
  // nothing to do here
}

asmo::MemoryLayout& asmo::NoMemoryLayout::Clone( void ) const
{
  return *new NoMemoryLayout(blockSize_);
}

size_type asmo::NoMemoryLayout::DoGetSegmentSize( void ) const
{
  return blockSize_;
}

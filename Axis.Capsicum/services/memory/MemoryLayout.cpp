#include "MemoryLayout.hpp"
#define MEMORY_WORD_LENGTH                  8ui64

namespace asmo = axis::services::memory;

asmo::MemoryLayout::MemoryLayout( void )
{
  // nothing to do here
}

asmo::MemoryLayout::~MemoryLayout( void )
{
  // nothing to do here
}

size_type asmo::MemoryLayout::GetSegmentSize( void ) const
{
  size_type sz = DoGetSegmentSize();
  return (sz + MEMORY_WORD_LENGTH - 1) & ~(MEMORY_WORD_LENGTH - 1);
}

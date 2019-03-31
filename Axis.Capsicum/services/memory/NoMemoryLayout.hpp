#pragma once
#include "MemoryLayout.hpp"

namespace axis { namespace services { namespace memory {

/**
 * Defines a simple memory layout comprised of a undistinguished chunk of data.
 */
class NoMemoryLayout : public MemoryLayout
{
public:
  /**
   * Constructor.
   *
   * @param blockSize Size of the block.
   */
  NoMemoryLayout(size_type blockSize);
  ~NoMemoryLayout(void);
  virtual MemoryLayout& Clone( void ) const;
private:
  virtual size_type DoGetSegmentSize( void ) const;
  size_type blockSize_;
};

} } } // namespace axis::services::memory

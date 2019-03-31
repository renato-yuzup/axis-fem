#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace services { namespace memory {

/**
 * Represents memory layout (disposition and alignment of data) of a 
 * memory block.
 */
class MemoryLayout
{
public:
  MemoryLayout(void);
  virtual ~MemoryLayout(void);

  /**
   * Makes a deep copy of this instance.
   *
   * @return A copy of this instance.
   */
  virtual MemoryLayout& Clone(void) const = 0;

  /**
   * Returns the total length, in bytes, occupied by a block with this layout in
   * memory.
   *
   * @return The block size.
   */
  size_type GetSegmentSize(void) const;  
private:

  /**
   * Calculates the total length, in bytes, occupied by a block with this 
   * layout in memory. It is not necessary to calculate memory alignment as
   * it is done by the base class.
   *
   * @return The block size.
   */
  virtual size_type DoGetSegmentSize(void) const = 0;  
};

} } } // namespace axis::services::memory

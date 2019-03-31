#pragma once
#include "foundation/Axis.SystemBase.hpp"
namespace axis { namespace services { namespace memory {

class MemoryLayout;

/**
 * Provides facilities to divide a memory range into several contiguous
 * cells with the same length and address each of them.
 */
class MemoryGrid
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] hostBlockArrayAddress     The base memory address of the 
   *                 range in host memory.
   * @param [in,out] mirroredBlockArrayAddress Corresponding base memory address
   *                 in the mirrored device.
   * @param layout                             Memory layout of the cells.
   * @param cellCount                          Number of cells.
   */
  MemoryGrid(void *hostBlockArrayAddress, void *mirroredBlockArrayAddress, 
    const MemoryLayout& layout, uint64 cellCount);
  ~MemoryGrid(void);

  /**
   * Returns cell address in host memory.
   *
   * @param index Zero-based index of the cell.
   *
   * @return The cell address.
   */
  void *GetHostCellAddress(uint64 index);

  /**
   * Returns cell address in host memory.
   *
   * @param index Zero-based index of the cell.
   *
   * @return The cell address.
   */
  const void *GetHostCellAddress(uint64 index) const;

  /**
   * Returns cell address in mirrored device memory.
   *
   * @param index Zero-based index of the cell.
   *
   * @return The cell address.
   */
  void *GetMirroredCellAddress(uint64 index);

  /**
   * Returns cell address in mirrored device memory.
   *
   * @param index Zero-based index of the cell.
   *
   * @return The cell address.
   */
  const void *GetMirroredCellAddress(uint64 index) const;

  /**
   * Returns cell count.
   *
   * @return The cell count.
   */
  uint64 GetCellCount(void) const;
private:
  void *hostBlockArrayAddress_;
  void *mirroredBlockArrayAddress_;
  MemoryLayout& layout_;
  uint64 cellCount_;
};

} } } // namespace axis::services::memory

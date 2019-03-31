#pragma once

namespace axis { namespace foundation { namespace mirroring { 

/**
 * Provides services to update data between host and external processing device.
 */
class MemoryMirror
{
public:
  MemoryMirror(void);
  virtual ~MemoryMirror(void);

  /**
   * Returns host-side base memory address.
   *
   * @return The base memory address.
   */
  virtual void *GetHostBaseAddress(void) const = 0;

  /**
   * Returns GPU-side base memory address.
   *
   * @return The base memory address.
   */
  virtual void *GetGPUBaseAddress(void) const = 0;

  /**
   * Allocates required memory in host and external processing device.
   */
  virtual void Allocate(void) = 0;

  /**
   * Deallocates claimed memory in host and external processing device.
   */
  virtual void Deallocate(void) = 0;

  /**
   * Copies data from host to external processing device memory.
   */
  virtual void Mirror(void) = 0;

  /**
   * Copies data from external processing device to host memory.
   */
  virtual void Restore(void) = 0;
};

} } } // namespace axis::services::memory

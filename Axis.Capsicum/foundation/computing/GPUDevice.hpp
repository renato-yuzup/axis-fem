#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "GPUCapability.hpp"
#include "nocopy.hpp"

namespace axis { namespace foundation { namespace computing {

class GPUQueue;

/**
 * Represents a supported GPGPU device available in the system.
**/
class GPUDevice
{
public:

  /**
   * Constructor.
   *
   * @param caps            The device capabilities.
   * @param totalMemorySize Device total memory size.
   * @param maxThreadCount  Maximum number of threads.
   * @param [in,out] queue  The device queue.
  **/
  GPUDevice(const GPUCapability& caps, uint64 totalMemorySize, 
    size_type maxThreadCount, GPUQueue& queue);
  ~GPUDevice(void);

  /**
   * Returns the corresponding work queue of this device.
   *
   * @return The queue.
  **/
  GPUQueue& GetQueue(void);

  /**
   * Returns the memory capacity of this device.
   *
   * @return The total device memory size.
  **/
  uint64 GetTotalMemorySize(void) const;

  /**
   * Returns the maximum number of simultaneous threads this device supports.
   *
   * @return The maximum thread count.
  **/
  size_type GetMaxThreadCount(void) const;

  /**
   * Returns the maximum number of simultaneous threads per block supported in
   * this device.
   *
   * @return The maximum thread count per block.
  **/
  size_type GetMaxThreadPerBlock(void) const;

  int GetMaxThreadDimX( void ) const;
  int GetMaxThreadDimY( void ) const;
  int GetMaxThreadDimZ( void ) const;
  int GetMaxGridDimX( void ) const;
  int GetMaxGridDimY( void ) const;
  int GetMaxGridDimZ( void ) const;

  /**
   * Sets this device as active so that further GPU operations are directed to it.
  **/
  void SetActive(void);

  /**
   * Allocates memory in the device.
   *
   * @param bytes Size of the memory block to allocate.
   *
   * @return A pointer to the device memory allocated.
  **/
  void *AllocateMemory(uint64 bytes);

  /**
   * Deallocates memory from the device.
   *
   * @param [in,out] memAddr Base address of the memory block to deallocate.
  **/
  void DeallocateMemory(void *memAddr);

  /**
   * Copies to device data allocated in the system memory.
   *
   * @param [in,out] targetDeviceMemAddress Base address of the device memory block where data will
   *                                        be written.
   * @param srcHostMemAddress               Base address of the host memory block.
   * @param blockSize                       Size of the block.
  **/
  void SendToDevice(void *targetDeviceMemAddress, const void *srcHostMemAddress, uint64 blockSize);

  /**
   * Transfers back to system memory data allocated in this device.
   *
   * @param [in,out] targetHostMemAddress Base address of the host memory block where data 
   *                                      will be written.
   * @param srcDeviceMemAddress           Base address of the device memory block.
   * @param blockSize                     Size of the block.
  **/
  void ReadFromDevice(void *targetHostMemAddress, const void *srcDeviceMemAddress, uint64 blockSize);

  /**
   * Blocks current thread until all tasks running on this device has finished.
  **/
  void Synchronize(void);

  /**
   * Marks a portion of memory as non-pageable so that it can be directly accessible by
   * this device.
   *
   * @param [in,out] targetMemory Base address of the memory block.
   * @param blockSize             Size of the block.
  **/
  void AttachMemory(void *targetMemory, uint64 blockSize);

  /**
   * Marks a portion of memory as pageable again. As a result, no device can directly access
   * the memory block.
   *
   * @param [in,out] targetMemory Base address of the memory block.
  **/
  void DetachMemory(void *targetMemory);

  /**
   * Returns the pointer used by the device to access a portion of memory 
   * previously marked as non-pageable.
   *
   * @param [in,out] attachedHostMemAddr  Non-pageable host memory address.
   *
   * @return  The device memory pointer.
  **/
  void *GetAttachedMemoryAddress(void *attachedHostMemAddr);
private:
  bool initialized_;
  uint64 memorySize_;
  size_type maxThreadCount_;
  GPUQueue *queue_;
  GPUCapability capabilities_;

  DISALLOW_COPY_AND_ASSIGN(GPUDevice);
};

} } } // namespace axis::foundation::computing

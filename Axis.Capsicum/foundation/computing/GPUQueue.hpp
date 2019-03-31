#pragma once
#include <cuda_runtime.h>

namespace axis { namespace foundation { namespace computing {

/**
 * Provides an interface to designate tasks to a GPU device.
**/
class GPUQueue
{
public:

  /**
   * Constructor.
   *
   * @param deviceIndex Unique identifier for this queue.
  **/
  GPUQueue(int deviceIndex);
  ~GPUQueue(void);

  /**
   * Returns a unique identifier for this queue among all 
   * available devices in this system.
   *
   * @return The queue index.
  **/
  int GetQueueIndex(void) const;

  /**
   * Returns a pointer to the queue of the associated device.
   *
   * @return The queue pointer.
  **/
  cudaStream_t GetQueue(void);

  /**
   * Returns a pointer to the queue of the associated device.
   *
   * @return The queue pointer.
  **/
  const cudaStream_t GetQueue(void) const;

  /**
   * Returns a pointer to the synchronization event associated with the device.
   *
   * @return The synchronization event pointer.
  **/
  cudaEvent_t GetEvent(void);

  /**
   * Returns a pointer to the synchronization event associated with the device.
   *
   * @return The synchronization event pointer.
  **/
  const cudaEvent_t GetEvent(void) const;

  /**
   * Blocks current thread until works in this queue are finished.
  **/
  void Synchronize(void);

  /**
   * Requests to insert a synchronization point in this device queue so that
   * subsequent calls to Synchronize() waits for task completion up to this
   * point.
  **/
  void RequestSynchronization(void);
private:
  void LazyInstantiate(void) const;
  int deviceIndex_;
  mutable cudaStream_t deviceStream_;
  mutable cudaEvent_t gpuEvent_;
};

} } } // namespace axis::foundation::computing

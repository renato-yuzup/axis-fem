#pragma once
#include <mutex>
#include "foundation/Axis.SystemBase.hpp"
#include "GPUCapability.hpp"

namespace axis { namespace foundation { namespace computing {

/**
 * Manages information about installed GPU cards.
 * This class is thread-safe.
**/
class GPUManager
{
public:
  GPUManager(void);
  ~GPUManager(void);

  /**
   * Rescans system for GPGPU compatible devices
   */
  void Rescan(void);

  /**
   * Returns how many GPGPU devices are available in the system.
   *
   * @return The available GPU count.
   */
  int GetAvailableGPUCount(void) const;

  /**
   * Queries GPU capabilities.
   *
   * @param gpuIndex Zero-based index of the GPU.
   *
   * @return The GPU capabilities.
   */
  GPUCapability GetGPUCapabilities(int gpuIndex) const;
private:
  mutable std::mutex mutex_;

  /**
   * Scans system for CUDA-compatible GPUs and populates GPU array
   * with each GPU index of a compatible device.
  **/
  void PopulateGPU(void);
  GPUCapability ReadGPUCapability(int index);

  int gpuCount_;
  int *realGpuIndex_;
  GPUCapability *gpuCapabilities_;
};

} } } // namespace axis::foundation::computing

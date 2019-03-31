#pragma once
#include "GPUManager.hpp"
#include <vector>

namespace axis { namespace foundation { namespace computing { 

class GPUDevice;

/**
 * Manager processing resources in the system.
 */
class ResourceManager
{
public:
  ResourceManager(void);
  ~ResourceManager(void);

  /**
   * Queries if CPU resources are available.
   *
   * @return true if CPU resource is available, false otherwise.
   */
  bool IsCPUResourcesAvailable(void) const;

  /**
   * Queries if GPU resources are available.
   *
   * @return true if GPU resource is available, false otherwise.
   */
  bool IsGPUResourcesAvailable(void) const;

  /**
   * Returns how many GPGPU devices are available.
   *
   * @return The available GPU count.
   */
  int GetAvailableGPUCount(void) const;

  /**
   * Allocates a GPU device for use by the program.
   *
   * @return Pointer to the GPUDevice representing the allocated device.
   * @sa GPUDevice
   */
  GPUDevice *AllocateGPU(void);

  /**
   * Releases the specified GPU device for other uses.
   *
   * @param [in,out] device The GPU device.
   *
   * @return true if it succeeds, false otherwise.
   */
  bool ReleaseGPU(GPUDevice& device);

  /**
   * Rescans the system for available processing devices.
   */
  void Rescan(void);
private:
  typedef std::vector<GPUDevice *> gpu_list;

  void InitGPUList(void);

  GPUManager manager_;
  gpu_list gpus_;
  gpu_list freeGpus_;
};

} } } // namespace axis::foundation::computing

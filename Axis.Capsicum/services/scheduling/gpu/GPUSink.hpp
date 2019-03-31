#pragma once
#include "services/messaging/CollectorEndpoint.hpp"

namespace axis { namespace services { namespace scheduling { namespace gpu {

/**
 * Provides error handling routines for GPU-related errors.
 */
class GPUSink : public axis::services::messaging::CollectorEndpoint
{
public:
  GPUSink(void);
  ~GPUSink(void);

  /**
   * Dispatches received message. If message is an error, it also toggles
   * failure, except when soft fail mode is enabled.
   *
   * @param [in,out] message The message.
   */
  void Notify(axis::services::messaging::EventMessage& message);

  /**
   * Triggers failure in application execution. Normally, it causes program
   * abortion.
   */
  virtual void Fail(void) = 0;

  /**
   * Toggles soft fail mode. When enabled, error messages are notified but does
   * not cause failure.
   *
   * @param state Soft fail mode state.
   */
  void ToggleSoftFail(bool state);
private:
  bool softFailEnabled_;
};

} } } } // namespace axis::services::scheduling::gpu

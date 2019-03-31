#pragma once
#include "foundation/Axis.Capsicum.hpp"
#include "services/messaging/CollectorHub.hpp"
#include "ExecutionRequest.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace scheduler {

/**
 * Receives execution requests and designates it to the adequate processing
 * queue for (possible immediately) execution.
 */
class AXISCAPSICUM_API ExecutionScheduler : 
  public axis::services::messaging::CollectorHub
{
public:
  ~ExecutionScheduler(void);

  /**
   * Submits an analysis for execution.
   *
   * @param [in,out] request Object containing job information.
   *
   * @return true if it succeeds, false if it fails.
  **/
  bool Submit(ExecutionRequest& request);

  /**
   * Returns current active execution scheduler.
   *
   * @return The active.
  **/
  static ExecutionScheduler& GetActive(void);
private:
  class Pimpl;

  ExecutionScheduler(void);

  Pimpl *pimpl_;
  static ExecutionScheduler *scheduler_;

  friend class Pimpl;

  ExecutionScheduler(const ExecutionScheduler&);
  ExecutionScheduler& operator =(const ExecutionScheduler&);
};

} } } } // namespace axis::application::scheduling::scheduler

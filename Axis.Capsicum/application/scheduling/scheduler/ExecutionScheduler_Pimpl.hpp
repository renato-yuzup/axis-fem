#pragma once
#include "ExecutionScheduler.hpp"
#include "application/scheduling/broker/SchedulerBroker.hpp"
#include "application/scheduling/broker/ProcessingQueueBroker.hpp"
#include "JobInspector.hpp"
#include "nocopy.hpp"
#include "foundation/computing/ResourceManager.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace scheduler {

class ExecutionScheduler::Pimpl
{
public:
  Pimpl(ExecutionScheduler& scheduler);
  ~Pimpl(void);

  void Error(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message);
  void Error(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message, const axis::String& str1);
  void Notify(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message);
  void Notify(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message, const axis::String& str1);
  void Warn(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message);
  void Warn(axis::services::messaging::Message::id_type messageId, 
    const axis::String& message, const axis::String& str1);

  axis::foundation::computing::ResourceManager *Manager;
  axis::application::scheduling::broker::SchedulerBroker *JobBroker;
  JobInspector *Inspector;
  axis::application::scheduling::broker::ProcessingQueueBroker *QueueBroker;
private:
  ExecutionScheduler& scheduler_;

  DISALLOW_COPY_AND_ASSIGN(Pimpl);
};

} } } } // namespace axis::application::scheduling::scheduler

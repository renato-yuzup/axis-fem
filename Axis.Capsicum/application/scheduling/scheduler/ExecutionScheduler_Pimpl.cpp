#include "ExecutionScheduler_Pimpl.hpp"
#include "foundation/capsicum_error.hpp"
#include "foundation/capsicum_info.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "services/messaging/WarningMessage.hpp"

namespace aasb = axis::application::scheduling::broker;
namespace aass = axis::application::scheduling::scheduler;
namespace asmm = axis::services::messaging;
namespace afc  = axis::foundation::computing;

aass::ExecutionScheduler::Pimpl::Pimpl(ExecutionScheduler& scheduler) : 
  scheduler_(scheduler)
{
  Manager = new afc::ResourceManager();
  JobBroker = new aasb::SchedulerBroker(*Manager);
  Inspector = new JobInspector();
  QueueBroker = new aasb::ProcessingQueueBroker(*Manager);
}

aass::ExecutionScheduler::Pimpl::~Pimpl(void)
{
  // nothing to do here
}

void aass::ExecutionScheduler::Pimpl::Error( asmm::Message::id_type messageId, 
  const axis::String& message )
{
  scheduler_.DispatchMessage(asmm::ErrorMessage(messageId, message));
}

void aass::ExecutionScheduler::Pimpl::Error( asmm::Message::id_type messageId, 
  const axis::String& message, const axis::String& str1 )
{
  String s = message;
  s.replace(_T("%1"), str1);
  scheduler_.DispatchMessage(asmm::ErrorMessage(messageId, s));
}

void aass::ExecutionScheduler::Pimpl::Notify( asmm::Message::id_type messageId, 
  const axis::String& message )
{
  scheduler_.DispatchMessage(asmm::InfoMessage(messageId, message));
}

void aass::ExecutionScheduler::Pimpl::Notify( asmm::Message::id_type messageId, 
  const axis::String& message, const axis::String& str1 )
{
  String s = message;
  s.replace(_T("%1"), str1);
  scheduler_.DispatchMessage(asmm::InfoMessage(messageId, s));
}

void aass::ExecutionScheduler::Pimpl::Warn( asmm::Message::id_type messageId, 
  const axis::String& message )
{
  scheduler_.DispatchMessage(asmm::WarningMessage(messageId, message));
}

void aass::ExecutionScheduler::Pimpl::Warn( asmm::Message::id_type messageId, 
  const axis::String& message, const axis::String& str1 )
{
  String s = message;
  s.replace(_T("%1"), str1);
  scheduler_.DispatchMessage(asmm::WarningMessage(messageId, s));
}

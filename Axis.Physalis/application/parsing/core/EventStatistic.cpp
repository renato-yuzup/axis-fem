#include "EventStatistic.hpp"
#include "EventStatistic_Pimpl.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aapc = axis::application::parsing::core;
namespace asmm = axis::services::messaging;

aapc::ParseContext::EventStatistic::EventStatistic( void )
{
  pimpl_ = new Pimpl();
  pimpl_->errorEventCount = 0;
  pimpl_->warningEventCount = 0;
  pimpl_->infoEventCount = 0;
  pimpl_->lastEvent = NULL;
}

aapc::ParseContext::EventStatistic::~EventStatistic( void )
{
  if (pimpl_->lastEvent != NULL)
  {
    pimpl_->lastEvent->Destroy();
    pimpl_->lastEvent = NULL;
  }
  delete pimpl_;
  pimpl_ = NULL;
}

long aapc::ParseContext::EventStatistic::GetErrorCount( void ) const
{
  return pimpl_->errorEventCount;
}

long aapc::ParseContext::EventStatistic::GetWarningCount( void ) const
{
  return pimpl_->warningEventCount;
}

long aapc::ParseContext::EventStatistic::GetInformationCount( void ) const
{
  return pimpl_->infoEventCount;
}

long aapc::ParseContext::EventStatistic::GetTotalEventCount( void ) const
{
  return pimpl_->infoEventCount + pimpl_->warningEventCount + pimpl_->errorEventCount;
}

bool aapc::ParseContext::EventStatistic::HasAnyEventRegistered( void ) const
{
  return GetTotalEventCount() > 0;
}

bool aapc::ParseContext::EventStatistic::HasErrorsRegistered( void ) const
{
  return GetErrorCount() > 0;
}

long aapc::ParseContext::EventStatistic::GetLastEventId( void ) const
{
  return pimpl_->lastEvent == NULL? 0 : pimpl_->lastEvent->GetId();
}

const asmm::EventMessage& aapc::ParseContext::EventStatistic::GetLastEvent( void ) const
{
  if (pimpl_->lastEvent == NULL)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return *pimpl_->lastEvent;
}

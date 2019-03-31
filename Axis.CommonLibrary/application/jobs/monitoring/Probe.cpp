#include "Probe.hpp"
#include "Probe_Pimpl.hpp"

using axis::application::jobs::monitoring::Probe;
using axis::services::messaging::ResultMessage;

axis::application::jobs::monitoring::Probe::Probe( void )
{
  pimpl_ = new Pimpl();
  pimpl_->probeEnabled = true;
}

Probe::~Probe(void)
{
  delete pimpl_;
  pimpl_ = NULL;
}

void Probe::ProcessResult( const ResultMessage& message )
{
  if (pimpl_->probeEnabled && IsInterestedInResult(message))
  {
    // clone message to avoid modifications
    ResultMessage& msg = (ResultMessage&)message.Clone();
    DoProcessResult(msg);
    msg.Destroy();
  }
}

void axis::application::jobs::monitoring::Probe::EnableProbe( void )
{
  pimpl_->probeEnabled = true;
}

void axis::application::jobs::monitoring::Probe::DisableProbe( void )
{
  pimpl_->probeEnabled = false;
}

bool axis::application::jobs::monitoring::Probe::IsEnabled( void ) const
{
  return pimpl_->probeEnabled;
}

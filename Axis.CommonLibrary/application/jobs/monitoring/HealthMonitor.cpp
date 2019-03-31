#include "HealthMonitor.hpp"
#include "HealthMonitor_Pimpl.hpp"
#include "Probe.hpp"
#include "Foundation/ArgumentException.hpp"

using axis::application::jobs::monitoring::HealthMonitor;
using axis::services::messaging::ResultMessage;

HealthMonitor::HealthMonitor(void)
{
  pimpl_ = new Pimpl();
}

HealthMonitor::~HealthMonitor(void)
{
  DestroyProbes();
  delete pimpl_;
  pimpl_ = NULL;
}

void axis::application::jobs::monitoring::HealthMonitor::AddProbe( Probe& probe )
{
  if (pimpl_->probes.find(&probe) != pimpl_->probes.end())
  {
    throw axis::foundation::ArgumentException(_T("probe"));
  }
  pimpl_->probes.insert(&probe);
}

void axis::application::jobs::monitoring::HealthMonitor::RemoveProbe( Probe& probe )
{
  if (pimpl_->probes.find(&probe) == pimpl_->probes.end())
  {
    throw axis::foundation::ArgumentException(_T("probe"));
  }
  pimpl_->probes.erase(&probe);
}

bool axis::application::jobs::monitoring::HealthMonitor::ContainsProbe( Probe& probe ) const
{
  return (pimpl_->probes.find(&probe) != pimpl_->probes.end());
}

void axis::application::jobs::monitoring::HealthMonitor::DestroyProbes( void )
{
  Pimpl::probe_set::iterator end = pimpl_->probes.end();
  for (Pimpl::probe_set::iterator it = pimpl_->probes.begin(); it != end; ++it)
  {
    Probe& probe = **it;
    probe.Destroy();
  }
}

void HealthMonitor::DoProcessResultMessage( ResultMessage& volatileMessage )
{
  Pimpl::probe_set::iterator end = pimpl_->probes.end();
  for (Pimpl::probe_set::iterator it = pimpl_->probes.begin(); it != end; ++it)
  {
    Probe& probe = **it;
    probe.ProcessResult(volatileMessage);
  }
}

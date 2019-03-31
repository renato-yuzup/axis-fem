#pragma once
#include "HealthMonitor.hpp"
#include <set>

namespace axis { namespace application { namespace jobs { namespace monitoring {

class HealthMonitor::Pimpl
{
public:
  typedef std::set<Probe *> probe_set;
  probe_set probes;
};

} } } } // namespace axis::application::jobs::monitoring

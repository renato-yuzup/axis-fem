#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "services/messaging/MessageListener.hpp"

namespace axis { namespace application { namespace jobs { namespace monitoring {

class Probe;

class AXISCOMMONLIBRARY_API HealthMonitor : public axis::services::messaging::MessageListener
{
public:
  HealthMonitor(void);
  ~HealthMonitor(void);
  void AddProbe(Probe& probe);
  void RemoveProbe(Probe& probe);
  bool ContainsProbe(Probe& probe) const;
  void DestroyProbes(void);
private:
  class Pimpl;
  virtual void DoProcessResultMessage( axis::services::messaging::ResultMessage& volatileMessage );
  Pimpl *pimpl_;
};

} } } } // namespace axis::application::jobs::monitoring

#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "services/messaging/ResultMessage.hpp"

namespace axis { namespace application { namespace jobs { namespace monitoring {

class AXISCOMMONLIBRARY_API Probe
{
public:
  Probe(void);
  virtual ~Probe(void);
  virtual void Destroy(void) const = 0;
  virtual bool IsInterestedInResult(const axis::services::messaging::ResultMessage& message) const = 0;
  void ProcessResult(const axis::services::messaging::ResultMessage& message);
  void EnableProbe(void);
  void DisableProbe(void);
  bool IsEnabled(void) const;
private:
  class Pimpl;
  virtual void DoProcessResult(const axis::services::messaging::ResultMessage& message) = 0;
  Pimpl *pimpl_;
};

} } } } // namespace axis::application::jobs::monitoring

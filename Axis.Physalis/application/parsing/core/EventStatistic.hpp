#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "ParseContext.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class AXISPHYSALIS_API ParseContext::EventStatistic
{
private:
  class Pimpl;
  Pimpl *pimpl_;
public:
  EventStatistic(void);
  ~EventStatistic(void);
  long GetErrorCount(void) const;
  long GetWarningCount(void) const;
  long GetInformationCount(void) const;
  long GetTotalEventCount(void) const;
  bool HasAnyEventRegistered(void) const;
  bool HasErrorsRegistered(void) const;
  long GetLastEventId(void) const;
  const axis::services::messaging::EventMessage& GetLastEvent(void) const;

  friend class ParseContext;
};

} } } } // namespace axis::application::parsing::core

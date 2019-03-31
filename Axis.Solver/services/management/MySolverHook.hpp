#pragma once
#include "services/management/SystemHook.hpp"

namespace axis { namespace services { namespace management {

class MySolverHook: public SystemHook
{
public:
  MySolverHook(void);
  virtual ~MySolverHook(void);
  virtual void ProcessMessage( int messageId, void *dataPtr );
};

} } } // namespace axis::services::management

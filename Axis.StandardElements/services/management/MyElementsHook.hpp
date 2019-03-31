#pragma once
#include "services/management/SystemHook.hpp"

namespace axis { namespace services { namespace management {

class MyElementsHook : public SystemHook
{
public:
  MyElementsHook(void);
  virtual ~MyElementsHook(void);
  virtual void ProcessMessage( int messageId, void *dataPtr );
};

} } } // namespace axis::services::management

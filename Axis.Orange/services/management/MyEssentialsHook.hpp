#pragma once
#include "services/management/SystemHook.hpp"

namespace axis { namespace services { namespace management {

class MyEssentialsHook : public SystemHook
{
public:
  MyEssentialsHook(void);
  virtual ~MyEssentialsHook(void);
  virtual void ProcessMessage( int messageId, void *dataPtr );
};

} } } // namespace axis::services::management

#pragma once
#include "services/management/SystemHook.hpp"

namespace axis { namespace services { namespace management {

class MyMaterialsHook : public SystemHook
{
public:
  MyMaterialsHook(void);
  virtual ~MyMaterialsHook(void);
  virtual void ProcessMessage( int messageId, void *dataPtr );
};

} } } // namespace axis::services::management

#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace services { namespace management {

/**
 * Represents an object which processes global system messages.
**/
class AXISSYSTEMBASE_API SystemHook
{
public:
  SystemHook(void);
  virtual ~SystemHook(void);

  /**
   * Processes system message.
   *
   * @param messageId         Identifier of the message.
   * @param [in,out] dataPtr  A generic data pointer associated with the message type. 
   *                          It can be null.
  **/
  virtual void ProcessMessage(int messageId, void *dataPtr) = 0;
};

} } } // namespace axis::services::management

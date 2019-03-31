#ifndef __MANAGER_MESSAGES_HPP
#define __MANAGER_MESSAGES_HPP

#include "message_base.hpp"

#define AXIS_LOG_MANAGER_POSTREGISTRATION_ERROR									_T("The module %1 cannot be registered due to an error. Check for troubleshooting information in the vendor's manual for this product.")
#define AXIS_LOG_MANAGER_REGISTRATIONDSECURITY_ERROR							_T("The module %1 cannot be registered because another module (%2) already claims the specified module path. Tried to register on %3. The current load protection mode doesn't allow this.")
#define AXIS_LOG_MANAGER_UNLOAD_ERROR											_T("The module %1 on %2 failed to unload due to an unexpected error. Execution will resume, but unexpected behavior might occur.")
#endif
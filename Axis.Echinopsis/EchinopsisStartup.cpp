#include "stdafx.h"
#include "EchinopsisStartup.hpp"
#include "services/management/EchinopsisPluginLoader.hpp"

axis::services::management::PluginLoader * AxisPluginLoader_GetLoader( void )
{
  return new axis::services::management::EchinopsisPluginLoader();
}

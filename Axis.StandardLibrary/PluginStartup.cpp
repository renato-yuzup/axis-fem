#include "stdafx.h"
#include "PluginStartup.hpp"
#include "services/management/SlPluginLoader.hpp"

axis::services::management::PluginLoader * AxisPluginLoader_GetLoader( void )
{
	return new axis::services::management::SlPluginLoader();
}

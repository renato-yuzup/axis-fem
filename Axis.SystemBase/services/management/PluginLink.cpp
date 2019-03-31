#include "PluginLink.hpp"


axis::services::management::PluginLink::PluginLink( const PluginInfo& pluginInfo, 
													const axis::String& pluginPath) :
_info(pluginInfo)
{
	_pluginPath = pluginPath;
}

axis::services::management::PluginInfo axis::services::management::PluginLink::GetPluginInformation( void ) const
{
	return _info;
}

axis::String axis::services::management::PluginLink::GetPluginPath( void ) const
{
	return _pluginPath;
}


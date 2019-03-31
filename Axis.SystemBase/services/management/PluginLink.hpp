#pragma once
#include "services/management/PluginInfo.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/collections/Collectible.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			/**************************************************************************************************
			 * <summary>	Describes a plugin established link into the application. </summary>
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API PluginLink : public axis::foundation::collections::Collectible
			{
			private:
				PluginInfo _info;
				axis::String _pluginPath;
			public:

				/**************************************************************************************************
				 * <summary>	Constructor. </summary>
				 *
				 * <param name="pluginInfo">		Plugin information object. </param>
				 * <param name="pluginPath">		Full pathname of the plugin file. </param>
				 **************************************************************************************************/
				PluginLink(const PluginInfo& pluginInfo, const axis::String& pluginPath);

				/**************************************************************************************************
				 * <summary>	Returns plugin information. </summary>
				 *
				 * <returns>	The plugin information. </returns>
				 **************************************************************************************************/
				PluginInfo GetPluginInformation(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the plugin full pathname. </summary>
				 *
				 * <returns>	The plugin path. </returns>
				 **************************************************************************************************/
				axis::String GetPluginPath(void) const;
			};
		}
	}
}

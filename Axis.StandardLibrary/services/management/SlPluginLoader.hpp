#pragma once
#include "services/management/PluginLoader.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			class SlPluginLoader : public axis::services::management::PluginLoader
			{
			public:
				virtual ~SlPluginLoader(void);

				virtual void StartPlugin( GlobalProviderCatalog& manager );

				virtual void Destroy( void ) const;

				virtual void UnloadPlugin( GlobalProviderCatalog& manager );

				virtual axis::services::management::PluginInfo GetPluginInformation( void ) const;
			};
		}
	}
}


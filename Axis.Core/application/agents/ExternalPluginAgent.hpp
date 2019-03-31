#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "services/configuration/ConfigurationScript.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/management/PluginLibrarian.hpp"
#include "services/io/DirectoryNavigator.hpp"
#include "foundation/collections/ObjectList.hpp"
#include "services/management/PluginLink.hpp"

namespace axis { namespace application { namespace agents {

class ExternalPluginAgent : public axis::services::messaging::CollectorHub
{
public:
	ExternalPluginAgent(void);
	~ExternalPluginAgent(void);
	void SetUp(axis::services::configuration::ConfigurationScript& script,
					axis::services::management::GlobalProviderCatalog& manager);
	void LoadSystemCustomizerPlugins(void);
	void LoadNonVolatilePlugins(void);
	void LoadVolatilePlugins(void);
	void EnumerateLoadedPlugins(void);

	/**************************************************************************************************
		* <summary>	Returns plugin link information. </summary>
		*
		* <param name="index">	Zero-based index of the plugin link to obtain information. </param>
		*
		* <returns>	The plugin link information. </returns>
		**************************************************************************************************/
	axis::services::management::PluginLink GetPluginLinkInfo(size_type index) const;
	size_type GetPluginCount(void) const;
private:
  void RunPluginDiscovery(axis::services::configuration::ConfigurationScript& script, 
    axis::services::management::PluginLayer targetLayer, 
    axis::services::management::GlobalProviderCatalog& manager);
  bool IsValidPluginFile( const axis::services::io::DirectoryNavigator& file ) const;
  bool IsValidPluginDescriptor( 
    axis::services::configuration::ConfigurationScript& pluginDescriptor ) const;
  void InterpretPluginDescriptor( axis::services::configuration::ConfigurationScript& pluginDescriptor, 
    axis::services::management::PluginLayer targetLayer );
  void LoadSinglePlugin( const axis::String& pluginPath, 
    axis::services::management::PluginLayer targetLayer );
  void ScanPluginLibrary( const axis::String& pluginFolderLocation, 
    axis::services::management::PluginLayer targetLayer, bool recursive );
  
  axis::services::configuration::ConfigurationScript *_script;
  axis::services::management::GlobalProviderCatalog *_manager;
  axis::services::management::PluginLibrarian *_librarian;				
  axis::foundation::collections::ObjectList *_pluginInfoList;
  size_type _pluginCount;
};		

} } } // namespace axis::application::agents

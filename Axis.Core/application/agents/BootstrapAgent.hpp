#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/configuration/ConfigurationScript.hpp"
#include "application/agents/CorePluginAgent.hpp"
#include "application/agents/ExternalPluginAgent.hpp"
#include "services/management/PluginLink.hpp"

namespace axis { namespace application { namespace agents {

/**********************************************************************************************//**
	* <summary> Expert responsible to execute the primary tasks to start up the program.</summary>
	*
	* <seealso cref="axis::services::messaging::CollectorHub"/>
	**************************************************************************************************/
class BootstrapAgent : public axis::services::messaging::CollectorHub
{
private:
	CorePluginAgent& _coreAgent;
	ExternalPluginAgent& _extensibilityAgent;
	axis::services::management::GlobalProviderCatalog *_manager;
	axis::services::configuration::ConfigurationScript *_configuration;

	bool DefinesPlugins( void ) const;
public:

	/**********************************************************************************************//**
		* <summary> Default constructor.</summary>
		**************************************************************************************************/
	BootstrapAgent(void);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	~BootstrapAgent(void);

	/**************************************************************************************************
		* <summary>	Sets configuration data to use to initiate bootstrap operation. </summary>
		*
		* <param name="configuration">	[in,out] The configuration script which contains bootstrap settings. </param>
		**************************************************************************************************/
	void SetUp(axis::services::configuration::ConfigurationScript& configuration);

	/**********************************************************************************************//**
		* <summary> Starts up the system module manager, initializes and
		* 			 registers all core system components and loads external
		* 			 plugins.</summary>
		**************************************************************************************************/
	void Run(void);

	/**********************************************************************************************//**
		* <summary> Returns the program module manager.</summary>
		*
		* <returns> The module manager.</returns>
		**************************************************************************************************/
	axis::services::management::GlobalProviderCatalog& GetModuleManager(void);

	/**********************************************************************************************//**
		* <summary> Returns the program module manager.</summary>
		*
		* <returns> The module manager.</returns>
		**************************************************************************************************/
	const axis::services::management::GlobalProviderCatalog& GetModuleManager(void) const;

	/**************************************************************************************************
		* <summary>	Returns plugin link information. </summary>
		*
		* <param name="index">	Zero-based index of the plugin link to obtain information. </param>
		*
		* <returns>	The plugin link information. </returns>
		**************************************************************************************************/
	axis::services::management::PluginLink GetPluginLinkInfo(size_type index) const;

	/**************************************************************************************************
		* <summary>	Returns how many plugin link was successfully established. </summary>
		*
		* <returns>	The plugin link count. </returns>
		**************************************************************************************************/
	size_type GetPluginLinkCount(void) const;
};		

} } } // namespace axis::application::agents

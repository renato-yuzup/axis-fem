#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "services/management/GlobalProviderCatalog.hpp"

namespace axis { namespace application { namespace agents {

/**********************************************************************************************//**
	* <summary> Expert responsible of starting up core system providers.</summary>
	*
	* <seealso cref="axis::services::messaging::CollectorHub"/>
	**************************************************************************************************/
class CorePluginAgent : public axis::services::messaging::CollectorHub
{
private:
	axis::services::management::GlobalProviderCatalog *_manager;

	/**********************************************************************************************//**
		* <summary> Registers the system feature locators in the module manager.</summary>
		*
		* <param name="manager"> [in,out] The program module manager.</param>
		**************************************************************************************************/
	void RegisterFeatureLocators(axis::services::management::GlobalProviderCatalog& manager);

	/**********************************************************************************************//**
		* <summary> Registers the system providers in the module manager.</summary>
		*
		* <param name="manager"> [in,out] The program module manager.</param>
		**************************************************************************************************/
	void RegisterInputProviders(axis::services::management::GlobalProviderCatalog& manager);

	/**********************************************************************************************//**
		* <summary> Registers the system factories in the system locators.</summary>
		*
		* <param name="manager"> [in,out] The module manager where system locators are registered.</param>
		**************************************************************************************************/
	void RegisterSystemFactories(axis::services::management::GlobalProviderCatalog& manager);
public:

	/**********************************************************************************************//**
		* <summary> Default constructor.</summary>
		**************************************************************************************************/
	CorePluginAgent(void);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	~CorePluginAgent(void);

	/**************************************************************************************************
		* <summary>	Sets up where modules should be registered and searched. </summary>
		*
		* <param name="manager">	[in,out] The program module manager. </param>
		**************************************************************************************************/
	void SetUp(axis::services::management::GlobalProviderCatalog& manager);

	/**********************************************************************************************//**
		* <summary> Registers the core system providers which are providers that cannot be 
		* 			 overridden by external plugins.</summary>
		**************************************************************************************************/
	void RegisterCoreProviders(void);

	/**********************************************************************************************//**
		* <summary> Registers the system customizable providers described in the
		* 			 module manager.</summary>
		**************************************************************************************************/
	void RegisterSystemCustomizableProviders(void);
};		

} } } // namespace axis::application::agents

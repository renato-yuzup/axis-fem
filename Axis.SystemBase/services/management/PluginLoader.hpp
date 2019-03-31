#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/management/PluginInfo.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			/**********************************************************************************************//**
			 * @brief	Represents a loader class which can be used to startup a
			 * 			plugin.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	27 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API PluginLoader
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 **************************************************************************************************/
				virtual ~PluginLoader(void);

				/**********************************************************************************************//**
				 * @brief	Starts a plugin.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param [in,out]	manager	The module manager which will manage plugin
				 * 					information and registration.
				 **************************************************************************************************/
				virtual void StartPlugin(GlobalProviderCatalog& manager) = 0;

				/**********************************************************************************************//**
				 * @brief	Unloads a plugin.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param [in,out]	manager	The manager.
				 **************************************************************************************************/
				virtual void UnloadPlugin(GlobalProviderCatalog& manager) = 0;

				/**************************************************************************************************
				 * <summary>	Returns general plugin information. </summary>
				 *
				 * <returns>	The plugin information. </returns>
				 **************************************************************************************************/
				virtual axis::services::management::PluginInfo GetPluginInformation(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void PluginLoader::Destroy(void) const = 0;
				 *
				 * @brief	Destroys this object.
				 *
				 * @author	Renato
				 * @date	06/05/2012
				 **************************************************************************************************/
				virtual void Destroy(void) const = 0;
			};
		}
	}
}


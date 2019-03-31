#pragma once
#include "foundation/Axis.Core.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/management/PluginInfo.hpp"

namespace axis { namespace services { namespace management {

class AXISCORE_API PluginConnector
{
public:
	/**********************************************************************************************//**
		* @enum	ErrorReason
		*
		* @brief	Reasons specifying why plugin could not be loaded.
		**************************************************************************************************/
	enum ErrorReason
	{
		///< Plugin module could not be loaded into the process address space.
		kInitializeError,

		///< No entry point funcion was found, probably meaning it is not a plugin.
		kEntryPointNotFound,

		///< Entry point found, but the loader function behaved in an unexpected way.
		kUnexpectedLoaderBehavior,

		///< When running, the loader threw an exception.
		kLoaderException,

		///< No errors happened until now.
		kNoError
	};

	virtual ~PluginConnector(void);
	virtual void LoadPlugin(void) = 0;
	virtual void UnloadPlugin(GlobalProviderCatalog& manager) = 0;
	virtual bool IsPluginLoaded(void) const = 0;

	/**********************************************************************************************//**
		* @fn	virtual bool PluginConnector::IsPluginReady(void) const = 0;
		*
		* @brief	Query if the plugin has already been loaded and registered all its functionalities.
		*
		* @author	Renato T. Yamassaki
		* @date	07 mai 2012
		*
		* @return	true if plugin is ready, false otherwise.
		**************************************************************************************************/
	virtual bool IsPluginReady(void) const = 0;
	virtual void RegisterPlugin(GlobalProviderCatalog& manager) = 0;
  virtual ErrorReason GetErrorCondition(void) const = 0;
	virtual void Destroy(void) const = 0;
	virtual axis::String GetFileName(void) const = 0;

	/**************************************************************************************************
		* <summary>	Returns plugin information. </summary>
		*
		* <returns>	The plugin information. </returns>
		**************************************************************************************************/
	virtual axis::services::management::PluginInfo GetPluginInformation(void) const = 0;
};

} } } // namespace axis::services::management

/// <summary>
/// Contains definition for the abstract class axis::services::management::Provider.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "GlobalProviderCatalog.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			/// <summary>
			/// Represents a feature provider.
			/// </summary>
			class AXISSYSTEMBASE_API Provider
			{
			public:
				virtual ~Provider(void);

				/**********************************************************************************************//**
				 * @fn	virtual const char Provider::*GetFeaturePath(void) const = 0;
				 *
				 * @brief	Returns the fully qualified feature path to which this
				 * 			provider is identified.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	11 abr 2012
				 *
				 * @return	null if it fails, else the feature path.
				 **************************************************************************************************/
				virtual const char *GetFeaturePath(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual const char Provider::*GetFeatureName(void) const = 0;
				 *
				 * @brief	Gets the feature name.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	10 mar 2011
				 *
				 * @return	The feature name.
				 **************************************************************************************************/
				virtual const char *GetFeatureName(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void Provider::PostProcessRegistration(axis::services::management::GlobalProviderCatalog& manager) = 0;
				 *
				 * @brief	Executes post processing tasks after registration. This
				 * 			operation should roll back every change made to the
				 * 			system if an error occurs.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	10 mar 2011
				 *
				 * @param [in]	manager	The module manager to which this provider was
				 * 						registered.
				 **************************************************************************************************/
				virtual void PostProcessRegistration(axis::services::management::GlobalProviderCatalog& manager);

				/**********************************************************************************************//**
				 * @fn	virtual void Provider::UnloadModule(axis::services::management::GlobalProviderCatalog& manager) = 0;
				 *
				 * @brief	Unloads this module.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	10 mar 2011
				 *
				 * @param [in]	manager	The module manager to which this provider is
				 * 						registered.
				 **************************************************************************************************/
				virtual void UnloadModule(axis::services::management::GlobalProviderCatalog& manager);
			};
		}
	}
}
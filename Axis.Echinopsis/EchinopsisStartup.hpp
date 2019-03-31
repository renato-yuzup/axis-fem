#pragma once
#include "foundation/axis.Echinopsis.hpp"
#include "services/management/PluginLoader.hpp"

/************************************************************************/
/* The plugin entry point function should be declared as a C function   */
/* so that the compiler do not decorate the function name.				*/
/************************************************************************/
extern "C" {
	/**********************************************************************************************//**
	 * @fn	AXISSTANDARDLIBRARY_API axis::services::management::PluginLoader *AxisPluginLoader_GetLoader(void);
	 *
	 * @brief	This function is the entry point used by the axis Plugin Loader to recognize this code
	 * 			library as a axis plugin. Also, this function will return a plugin loader object which
	 * 			will be used to register the functionalities and query information about this library.
	 *
	 * @author	Renato
	 * @date	06/05/2012
	 *
	 * @return	null if it fails, else.
	 **************************************************************************************************/

	AXISECHINOPSIS_API axis::services::management::PluginLoader *AxisPluginLoader_GetLoader(void);
};
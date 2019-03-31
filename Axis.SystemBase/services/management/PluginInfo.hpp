#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			/**************************************************************************************************
			 * <summary>	Stores information about a plugin. </summary>
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API PluginInfo
			{
			private:
				axis::String _name;
				axis::String _description;
				axis::String _internalName;
				axis::String _author;
				axis::String _copyright;
				axis::String _versionString;
				int _major, _minor, _revision, _build;
			public:

				/**************************************************************************************************
				 * <summary>	Constructor. </summary>
				 *
				 * <param name="pluginName">				 	Name of the plugin. </param>
				 * <param name="pluginDescription">			 	Information describing the plugin. </param>
				 * <param name="internalName">				 	Internal name of the plugin. </param>
				 * <param name="authorName">				 	Name of the author. </param>
				 * <param name="copyrightNotice">			 	The copyright notice. </param>
				 * <param name="pluginMajorVersionNumber">   	The plugin major version number. </param>
				 * <param name="pluginMinorVersionNumber">   	The plugin minor version number. </param>
				 * <param name="pluginRevisionVersionNumber">	The plugin revision version number. </param>
				 * <param name="pluginBuildVersionNumber">   	The plugin build version number. </param>
				 * <param name="versionReleaseString">		 	The version release string. </param>
				 **************************************************************************************************/
				PluginInfo(const axis::String& pluginName, 
						   const axis::String& pluginDescription, 
						   const axis::String& internalName, 
						   const axis::String& authorName, 
						   const axis::String& copyrightNotice, 
						   int pluginMajorVersionNumber, 
						   int pluginMinorVersionNumber, 
						   int pluginRevisionVersionNumber, 
						   int pluginBuildVersionNumber, 
						   const axis::String& versionReleaseString
						   );

				/**********************************************************************************************//**
				 * @brief	Returns the plugin name.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin name.
				 **************************************************************************************************/
				axis::String GetPluginName(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin description.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin description.
				 **************************************************************************************************/
				axis::String GetPluginDescription(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin internal name.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin internal name.
				 **************************************************************************************************/
				axis::String GetPluginInternalName(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin author.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin author.
				 **************************************************************************************************/
				axis::String GetPluginAuthor(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin copyright notice.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin copyright notice.
				 **************************************************************************************************/
				axis::String GetPluginCopyrightNotice(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin version release string.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin version release string.
				 **************************************************************************************************/
				axis::String GetPluginVersionReleaseString(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin major version.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin major version.
				 **************************************************************************************************/
				int GetPluginMajorVersion(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin minor version.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin minor version.
				 **************************************************************************************************/
				int GetPluginMinorVersion(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin revision number.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin revision version.
				 **************************************************************************************************/
				int GetPluginRevisionVersion(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the plugin build number.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	The plugin build version.
				 **************************************************************************************************/
				int GetPluginBuildVersion(void) const;
			};
		}
	}
}


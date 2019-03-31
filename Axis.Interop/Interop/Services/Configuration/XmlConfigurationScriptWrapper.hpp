#pragma once
#include "services/configuration/ConfigurationScript.hpp"

namespace axis
{
	namespace Interop
	{
		namespace Services
		{
			namespace Configuration
			{
				ref class XmlConfigurationScriptWrapper
				{
				private:
					System::String ^_fileName;
					axis::services::configuration::ConfigurationScript *_xmlScript;
				public:

					/**********************************************************************************************//**
					 * @fn	XmlConfigurationScriptWrapper::XmlConfigurationScriptWrapper(System::String ^fileName);
					 *
					 * @brief	Creates a new XML script read from a file in disk.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	27 mar 2012
					 *
					 * @param	fileName	Path of the file.
					 **************************************************************************************************/
					XmlConfigurationScriptWrapper(System::String ^fileName);

					/**********************************************************************************************//**
					 * @fn	System::String XmlConfigurationScriptWrapper::GetFileName(void);
					 *
					 * @brief	Gets the script file path.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	27 mar 2012
					 *
					 * @return	The file path.
					 **************************************************************************************************/
					System::String ^GetFileName(void);

					/**********************************************************************************************//**
					 * @fn	XmlConfigurationScriptWrapper::~XmlConfigurationScriptWrapper(void);
					 *
					 * @brief	Destructor of this class. Called on all other cases besides garbage collection.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	27 mar 2012
					 **************************************************************************************************/
					~XmlConfigurationScriptWrapper(void);

					/**********************************************************************************************//**
					 * @fn	XmlConfigurationScriptWrapper::!XmlConfigurationScriptWrapper(void);
					 *
					 * @brief	Finaliser of this class. Called when doing garbage collection.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	27 mar 2012
					 **************************************************************************************************/
					!XmlConfigurationScriptWrapper(void);
				internal:

					/**********************************************************************************************//**
					 * @fn	axis::services::configuration::XmlConfigurationScript XmlConfigurationScriptWrapper::*GetWrappedScript(void);
					 *
					 * @brief	Gets the wrapped unmanaged script configuration object held by this instance.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	27 mar 2012
					 *
					 * @return	A pointer to the wrapped object.
					 **************************************************************************************************/
					axis::services::configuration::ConfigurationScript *GetWrappedScript(void);
				};			
			}
		}
	}
}


#pragma once
#include "services/messaging/CollectorHub.hpp"
#include "services/configuration/ConfigurationScript.hpp"

namespace axis { namespace application { namespace agents {

/**********************************************************************************************//**
	* <summary> Provides lookup and parsing of configuration data stored
	* 			 in a file.</summary>
	*
	* <seealso cref="axis::services::messaging::CollectorHub"/>
	**************************************************************************************************/
class ConfigurationLoaderAgent : public axis::services::messaging::CollectorHub
{
private:
	axis::String _configurationFileLocation;
	axis::services::configuration::ConfigurationScript *_script;
public:

	/**********************************************************************************************//**
		* <summary> Default constructor.</summary>
		**************************************************************************************************/
	ConfigurationLoaderAgent(void);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	~ConfigurationLoaderAgent(void);

	/**********************************************************************************************//**
		* <summary> Sets the filename to lookup when parsing configuration.</summary>
		*
		* <param name="configurationFileLocation"> The configuration file
		* 											location.</param>
		**************************************************************************************************/
	void SetConfigurationLocation(const axis::String& configurationFileLocation);

	/**************************************************************************************************
		* <summary>	Returns the pathname for the configuration file. </summary>
		*
		* <returns>	The configuration pathname. </returns>
		**************************************************************************************************/
	axis::String GetConfigurationLocation(void) const;


	/**********************************************************************************************//**
		* <summary> Loads the configuration from the specified file.</summary>
		**************************************************************************************************/
	void LoadConfiguration(void);

	/**********************************************************************************************//**
		* <summary> Returns the parsed configuration.</summary>
		*
		* <returns> The parsed configuration.</returns>
		**************************************************************************************************/
	axis::services::configuration::ConfigurationScript& GetConfiguration(void) const;
};		

} } } // namespace axis::application::agents

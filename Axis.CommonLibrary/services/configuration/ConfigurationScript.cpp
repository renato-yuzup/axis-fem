#include "ConfigurationScript.hpp"
#include "XmlConfigurationScript.hpp"

axis::services::configuration::ConfigurationScript::~ConfigurationScript( void )
{
	// do nothing
}

axis::services::configuration::ConfigurationScript& axis::services::configuration::ConfigurationScript::ReadFromXml( const axis::String& filename )
{
	return axis::services::configuration::XmlConfigurationScript::ReadFromFile(filename);
}
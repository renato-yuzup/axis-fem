#include "ScriptFactory.hpp"
#include "XmlConfigurationScript.hpp"

axis::services::configuration::ScriptFactory::ScriptFactory( void )
{
	// do nothing
}

axis::services::configuration::ScriptFactory::~ScriptFactory( void )
{
	// do nothing
}

axis::services::configuration::ConfigurationScript& axis::services::configuration::ScriptFactory::ReadFromXmlFile( const axis::String& fileName )
{
	return XmlConfigurationScript::ReadFromFile(fileName);
}
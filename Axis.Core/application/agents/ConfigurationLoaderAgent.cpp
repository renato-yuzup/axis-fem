#include "ConfigurationLoaderAgent.hpp"
#include "services/configuration/XmlConfigurationScript.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/definitions/ConfigurationFileDescriptor.hpp"
#include "foundation/ConfigurationSectionNotFoundException.hpp"

namespace aaa = axis::application::agents;
namespace asc = axis::services::configuration;
namespace afd = axis::foundation::definitions;

aaa::ConfigurationLoaderAgent::ConfigurationLoaderAgent( void )
{
	_script = NULL;
}

aaa::ConfigurationLoaderAgent::~ConfigurationLoaderAgent( void )
{
	if (_script != NULL) _script->Destroy();
	_script = NULL;
}

void aaa::ConfigurationLoaderAgent::SetConfigurationLocation( 
  const axis::String& configurationFileLocation )
{
	_configurationFileLocation = configurationFileLocation;
}

axis::String aaa::ConfigurationLoaderAgent::GetConfigurationLocation( void ) const
{
	return _configurationFileLocation;
}

void aaa::ConfigurationLoaderAgent::LoadConfiguration( void )
{
	if (_configurationFileLocation.empty())
	{
		throw axis::foundation::InvalidOperationException(
      _T("Cannot parse file because its location has not been specified."));
	}
	asc::ConfigurationScript& script = 
    asc::XmlConfigurationScript::ReadFromXml(_configurationFileLocation);
	if (_script != NULL)
	{
		_script->Destroy();
	}
	_script = &script;
	// check that we have a valid root section
	if (_script->GetSectionName() != afd::ConfigurationFileDescriptor::RootSectionName)
	{
		_script->Destroy();
		_script = NULL;
		throw axis::foundation::ConfigurationSectionNotFoundException(
      _T("The specified file doesn't seem to be a valid configuration script."));
	}
}

asc::ConfigurationScript& aaa::ConfigurationLoaderAgent::GetConfiguration( void ) const
{
	if (_script == NULL)
	{
		throw axis::foundation::InvalidOperationException(
      _T("configuration must be parsed prior to this operation."));
	}
	return *_script;
}

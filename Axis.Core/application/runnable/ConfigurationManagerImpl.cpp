#include "ConfigurationManagerImpl.hpp"
#include "services/io/FileSystem.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aar = axis::application::runnable;
namespace asc = axis::services::configuration;
namespace asio = axis::services::io;

aar::ConfigurationManagerImpl::ConfigurationManagerImpl( void )
{
	_configAgent.SetConfigurationLocation(asio::FileSystem::GetDefaultConfigurationFileLocation());
}

aar::ConfigurationManagerImpl::~ConfigurationManagerImpl( void )
{
  // nothing to do here
}

axis::String aar::ConfigurationManagerImpl::GetConfigurationScriptPath( void ) const
{
	return _configAgent.GetConfigurationLocation();
}

void aar::ConfigurationManagerImpl::SetConfigurationScriptPath( const axis::String& configPath )
{
	_configAgent.SetConfigurationLocation(configPath);
}

void aar::ConfigurationManagerImpl::LoadConfiguration( void )
{
	_configAgent.LoadConfiguration();
}

asc::ConfigurationScript& aar::ConfigurationManagerImpl::GetConfiguration( void )
{
	return _configAgent.GetConfiguration();
}

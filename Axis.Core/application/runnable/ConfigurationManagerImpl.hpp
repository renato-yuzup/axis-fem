#pragma once
#include "ConfigurationManager.hpp"
#include "application/agents/ConfigurationLoaderAgent.hpp"

namespace axis { namespace application { namespace runnable {

class ConfigurationManagerImpl : public ConfigurationManager
{
public:
	ConfigurationManagerImpl(void);
	~ConfigurationManagerImpl(void);
	virtual axis::String GetConfigurationScriptPath( void ) const;
	virtual void SetConfigurationScriptPath( const axis::String& configPath );
	virtual void LoadConfiguration( void );
	virtual axis::services::configuration::ConfigurationScript& GetConfiguration( void );
private:
	axis::application::agents::ConfigurationLoaderAgent _configAgent;
};

} } } // namespace axis::application::runnable

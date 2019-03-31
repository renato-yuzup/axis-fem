#pragma once
#include "foundation/Axis.Core.hpp"
#include "AxisString.hpp"
#include "services/configuration/ConfigurationScript.hpp"

namespace axis { namespace application { namespace runnable {

class AXISCORE_API ConfigurationManager 
{
public:				
	virtual ~ConfigurationManager(void);

	virtual axis::String GetConfigurationScriptPath(void) const = 0;
	virtual void SetConfigurationScriptPath(const axis::String& configPath) = 0;

	virtual void LoadConfiguration(void) = 0;
	virtual axis::services::configuration::ConfigurationScript& GetConfiguration(void) = 0;
};

} } } // namespace axis::application::runnable

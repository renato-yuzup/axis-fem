#pragma once
#include "PluginConnector.hpp"
#include "services/management/PluginLoader.hpp"

namespace axis { namespace services { namespace management {

class WindowsPluginConnector : public PluginConnector
{
public:
	WindowsPluginConnector(const axis::String& dllFileName);
	~WindowsPluginConnector(void);
	virtual void LoadPlugin( void );
	virtual void UnloadPlugin(GlobalProviderCatalog& manager);
	virtual bool IsPluginLoaded( void ) const;
	virtual void RegisterPlugin( GlobalProviderCatalog& manager );
	virtual ErrorReason GetErrorCondition( void ) const;
	virtual void Destroy( void ) const;
	virtual axis::String GetFileName( void ) const;
	virtual bool IsPluginReady( void ) const;
	virtual axis::services::management::PluginInfo GetPluginInformation( void ) const;
private:
	ErrorReason _lastError;
	axis::String _dllPath;
	void *_moduleHandle;
	axis::services::management::PluginLoader *_loader;
	bool _pluginLoaded;
	bool _isRegistered;
};

} } } // namespace axis::services::management

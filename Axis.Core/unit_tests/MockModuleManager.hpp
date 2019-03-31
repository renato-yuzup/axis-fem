#pragma once
#include "services/management/GlobalProviderCatalog.hpp"

class MockModuleManager : public axis::services::management::GlobalProviderCatalog
{
public:
	MockModuleManager(void);
	~MockModuleManager(void);

	virtual axis::services::management::Provider& GetProvider( const char *providerPath ) const;

	virtual void RegisterProvider( axis::services::management::Provider& provider );

	virtual bool ExistsProvider( const char *providerPath ) const;

	virtual void UnregisterProvider( axis::services::management::Provider& provider );
};


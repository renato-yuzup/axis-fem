#pragma once
#include "ProviderProxy.hpp"

namespace axis { namespace services { namespace management {

class SystemModuleProxy : public ProviderProxy
{
public:
	SystemModuleProxy(Provider& provider, bool isBootstrapModule, bool isCustom);
	virtual ~SystemModuleProxy(void);
	virtual bool IsSystemModule(void) const;
	virtual bool IsBootstrapModule(void) const;
	virtual bool IsNonVolatileUserModule(void) const;
	virtual bool IsUserModule(void) const;
	virtual Provider& GetProvider(void) const;
	virtual bool IsCustomSystemModule( void ) const;
private:
	bool _isBootstrap;
	bool _isCustom;
	Provider& _provider;
};		

} } } // namespace axis::services::management

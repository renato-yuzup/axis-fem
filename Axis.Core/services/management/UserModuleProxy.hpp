#pragma once
#include "ProviderProxy.hpp"

namespace axis { namespace services { namespace management {

class UserModuleProxy : public ProviderProxy
{
public:
	UserModuleProxy(Provider& provider, bool isNonVolatile);
	virtual ~UserModuleProxy(void);
	virtual bool IsBootstrapModule(void) const;
	virtual bool IsNonVolatileUserModule(void) const;
	virtual bool IsUserModule(void) const;
	virtual Provider& GetProvider(void) const;
	virtual bool IsCustomSystemModule( void ) const;
	virtual bool IsSystemModule( void ) const;
private:
	bool _isNonVolatile;
	bool _isWeakModule;
	Provider& _provider;
};		

} } } // namespace axis::services::management

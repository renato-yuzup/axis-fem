#pragma once
#include "services/management/Provider.hpp"

namespace axis { namespace services { namespace management { 

class ProviderProxy
{
public:
	virtual ~ProviderProxy(void);
	virtual bool IsCustomSystemModule(void) const = 0;
	virtual bool IsSystemModule(void) const = 0;
	virtual bool IsBootstrapModule(void) const = 0;
	virtual bool IsNonVolatileUserModule(void) const = 0;
	virtual bool IsUserModule(void) const = 0;
	virtual Provider& GetProvider(void) const = 0;
};		

} } } // namespace axis::services::management

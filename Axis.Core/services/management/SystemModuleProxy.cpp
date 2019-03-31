#include "SystemModuleProxy.hpp"

namespace asmg = axis::services::management;

asmg::SystemModuleProxy::SystemModuleProxy( Provider& provider, bool isBootstrapModule, 
                                            bool isCustom ) : _provider(provider)
{
	_isBootstrap = isBootstrapModule;
	_isCustom = isCustom;
}

asmg::SystemModuleProxy::~SystemModuleProxy( void )
{
	// nothing to do here
}

bool asmg::SystemModuleProxy::IsSystemModule( void ) const
{
	return true;
}

bool asmg::SystemModuleProxy::IsBootstrapModule( void ) const
{
	return _isBootstrap;
}

bool asmg::SystemModuleProxy::IsNonVolatileUserModule( void ) const
{
	return false;
}

bool asmg::SystemModuleProxy::IsUserModule( void ) const
{
	return false;
}

asmg::Provider& asmg::SystemModuleProxy::GetProvider( void ) const
{
	return _provider;
}

bool asmg::SystemModuleProxy::IsCustomSystemModule( void ) const
{
	return _isCustom;
}

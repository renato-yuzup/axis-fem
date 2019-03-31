#include "UserModuleProxy.hpp"

namespace asmg = axis::services::management;

asmg::UserModuleProxy::UserModuleProxy( Provider& provider, bool isNonVolatile ) : 
_provider(provider)
{
	_isNonVolatile = isNonVolatile;
}

asmg::UserModuleProxy::~UserModuleProxy( void )
{
	// nothing to do here
}

bool asmg::UserModuleProxy::IsSystemModule( void ) const
{
	return false;
}

bool asmg::UserModuleProxy::IsBootstrapModule( void ) const
{
	return false;
}

bool asmg::UserModuleProxy::IsNonVolatileUserModule( void ) const
{
	return _isNonVolatile;
}

bool asmg::UserModuleProxy::IsUserModule( void ) const
{
	return true;
}

asmg::Provider& asmg::UserModuleProxy::GetProvider( void ) const
{
	return _provider;
}
bool asmg::UserModuleProxy::IsCustomSystemModule( void ) const
{
	return false;
}

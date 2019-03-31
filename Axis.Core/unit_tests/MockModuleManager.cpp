#if defined DEBUG || defined _DEBUG

#include "MockModuleManager.hpp"
#include <string>
#include "application/factories/elements/NodeFactory.hpp"
#include "services/management/ServiceLocator.hpp"

MockModuleManager::MockModuleManager(void)
{
}

MockModuleManager::~MockModuleManager(void)
{
}

axis::services::management::Provider& MockModuleManager::GetProvider( const char *providerPath ) const
{
	if(strcmp(axis::services::management::ServiceLocator::GetNodeFactoryPath(), providerPath) == 0)
	{
		return *new axis::application::factories::elements::NodeFactory();
	}
	throw std::exception("Provider not found.");
}

void MockModuleManager::RegisterProvider( axis::services::management::Provider& provider )
{
	throw std::exception("The method or operation is not implemented.");
}

bool MockModuleManager::ExistsProvider( const char *providerPath ) const
{
	throw std::exception("The method or operation is not implemented.");
}

void MockModuleManager::UnregisterProvider( axis::services::management::Provider& provider )
{
	throw std::exception("The method or operation is not implemented.");
}

#endif

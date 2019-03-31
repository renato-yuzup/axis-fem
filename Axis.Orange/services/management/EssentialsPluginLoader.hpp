#pragma once
#include <set>
#include "services/management/PluginLoader.hpp"
#include "application/factories/boundary_conditions/ConstraintFactory.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "MyEssentialsHook.hpp"

namespace axis { namespace services { namespace management {

class EssentialsPluginLoader : public axis::services::management::PluginLoader
{
public:
	EssentialsPluginLoader(void);
	virtual ~EssentialsPluginLoader(void);
	virtual void StartPlugin( GlobalProviderCatalog& manager );
	virtual void UnloadPlugin( GlobalProviderCatalog& manager );
	virtual void Destroy( void ) const;
	virtual axis::services::management::PluginInfo GetPluginInformation( void ) const;
private:
	typedef std::set<axis::application::factories::boundary_conditions::ConstraintFactory *> 
    constraint_factory_set;
  typedef std::set<axis::application::factories::parsers::BlockProvider *> provider_set;

	void InitFactories(void);
	void DestroyFactories(void);

	constraint_factory_set constraintFactories_;
  provider_set curveProviders_;
  provider_set loadProviders_;
  MyEssentialsHook hook_;
};

} } } // namespace axis::services::management

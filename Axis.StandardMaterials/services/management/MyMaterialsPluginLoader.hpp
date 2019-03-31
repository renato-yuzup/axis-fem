#pragma once
#include <set>
#include "services/management/PluginLoader.hpp"
#include "application/factories/materials/MaterialFactory.hpp"
#include "MyMaterialsHook.hpp"

namespace axis { namespace services { namespace management {

class MyMaterialsPluginLoader : public axis::services::management::PluginLoader
{
public:
	MyMaterialsPluginLoader(void);
	virtual ~MyMaterialsPluginLoader(void);
	virtual void StartPlugin( GlobalProviderCatalog& manager );
	virtual void UnloadPlugin( GlobalProviderCatalog& manager );
	virtual void Destroy( void ) const;
	virtual axis::services::management::PluginInfo GetPluginInformation( void ) const;
private:
	typedef std::set<axis::application::factories::materials::MaterialFactory *> factory_set;
	void LoadMaterialFactories(void);
	void UnloadMaterialFactories(void);
	factory_set _materialFactories;
  MyMaterialsHook hook_;
};

} } } //namespace axis::services::management

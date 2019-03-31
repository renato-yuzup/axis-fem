#pragma once
#include <set>
#include "application/factories/parsers/ElementParserFactory.hpp"
#include "application/locators/ElementParserLocator.hpp"
#include "services/management/PluginLoader.hpp"
#include "MyElementsHook.hpp"

namespace axis { namespace services { namespace management {

class MyElementsPluginLoader : public axis::services::management::PluginLoader
{
public:
	MyElementsPluginLoader(void);
	virtual ~MyElementsPluginLoader(void);
	virtual void StartPlugin( GlobalProviderCatalog& manager );
	virtual void UnloadPlugin( GlobalProviderCatalog& manager );
	virtual void Destroy( void ) const;
	virtual axis::services::management::PluginInfo GetPluginInformation( void ) const;
private:
  typedef std::set<axis::application::factories::parsers::ElementParserFactory *> factory_set;
  void InitFactories(axis::application::locators::ElementParserLocator& locator);
  void DestroyFactories(void);
  factory_set _ourFactories;
  MyElementsHook hook_;
};

} } } // namespace axis::services::management

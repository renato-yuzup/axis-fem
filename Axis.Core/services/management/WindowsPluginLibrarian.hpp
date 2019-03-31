#pragma once
#include "services/management/PluginLibrarian.hpp"
#include <map>
#include <set>

namespace axis { namespace services { namespace management {

class WindowsPluginLibrarian : public PluginLibrarian
{
public:
	class IteratorLogicImpl : public IteratorLogic
	{
	public:
    typedef std::map<axis::String, PluginConnector *> global_map;

    IteratorLogicImpl(const global_map::iterator& it);
		virtual bool operator ==(const IteratorLogic& logic) const;
		virtual bool operator !=(const IteratorLogic& logic) const;
		virtual void Destroy(void) const;
		virtual IteratorLogic& Clone(void) const;
		virtual IteratorLogic& GoNext(void); 
		virtual PluginConnector& GetItem(void) const;
  private:
    global_map::iterator _it;
	};

	WindowsPluginLibrarian(void);
	virtual ~WindowsPluginLibrarian(void);
	virtual void Destroy( void ) const;
	virtual const axis::services::management::PluginConnector& AddConnector( 
    const axis::String& pluginFileName, PluginLayer operationLayer );
	virtual void UnloadPluginLayer( PluginLayer layer, GlobalProviderCatalog& manager );
	virtual void LoadPluginLayer( PluginLayer layer, GlobalProviderCatalog& manager );
	virtual PluginConnector& GetPluginConnector( const axis::String& pluginFileName ) const;
	virtual bool ContainsPlugin( const axis::String& pluginFileName ) const;
	virtual bool IsPluginLoaded( const axis::String& pluginFileName ) const;
	virtual unsigned int PluginCount( void ) const;
	virtual Iterator GetIterator( void ) const;
private:
  typedef std::map<axis::String, PluginConnector *> global_map;
  typedef std::map<axis::String, bool> load_map;
  typedef std::set<PluginConnector *> plugin_set;
  typedef std::map<PluginLayer, plugin_set *> layer_set;

  global_map *_plugins;
  load_map *_loadStatus;
  layer_set *_layers;
};		

} } } // namespace axis::services::management

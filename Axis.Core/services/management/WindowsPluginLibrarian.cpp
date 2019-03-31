#include <windows.h>

#include "WindowsPluginLibrarian.hpp"
#include "WindowsPluginConnector.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidPluginException.hpp"
#include "foundation/BadPluginException.hpp"

namespace asmg = axis::services::management;

asmg::WindowsPluginLibrarian::WindowsPluginLibrarian( void )
{
	_plugins = new global_map();
	_loadStatus = new load_map();
	_layers = new layer_set();
}

asmg::WindowsPluginLibrarian::~WindowsPluginLibrarian( void )
{
	delete _plugins;
	delete _loadStatus;

	// it is dangerous to destroy this object if the plugins were not unloaded,
	// but anyways...
	layer_set::iterator end = _layers->end();
	for (layer_set::iterator it = _layers->begin(); it != end; ++it)
	{
		delete it->second;
	}
	delete _layers;
}

void asmg::WindowsPluginLibrarian::Destroy( void ) const
{
	delete this;
}

const asmg::PluginConnector& asmg::WindowsPluginLibrarian::AddConnector( 
  const axis::String& pluginFileName, PluginLayer operationLayer )
{
	// normalize file name so that we can make more precise comparisons later
	axis::String s = pluginFileName;
	s.trim().to_lower_case();
	if (ContainsPlugin(s)) 
	{
		throw axis::foundation::ArgumentException();
	}
	// create a new connector and try to initialize
	PluginConnector *c = new WindowsPluginConnector(s);
	c->LoadPlugin();
	if (c->GetErrorCondition() != PluginConnector::kNoError)
	{	// an error occurred -- destroy connector
		PluginConnector::ErrorReason reason = c->GetErrorCondition();
		c->Destroy();

		switch (reason)
		{
		case PluginConnector::kInitializeError:				// not a valid DLL
		case PluginConnector::kEntryPointNotFound:			// it is a DLL, but not a plugin
			throw axis::foundation::InvalidPluginException();
			break;
		case PluginConnector::kLoaderException:				// it is a plugin, but an error occurred
		case PluginConnector::kUnexpectedLoaderBehavior:		// it seems to be a plugin, but it is bad coded
			throw axis::foundation::BadPluginException();
			break;
		}
	}
	// add connector to indexes
	(*_plugins)[s] = c;
	(*_loadStatus)[s] = false;
	plugin_set *set;
	layer_set::iterator it = _layers->find(operationLayer);
	if (it != _layers->end())
	{
		set = it->second;
	}
	else
	{	// set doesn't exist -- create
		set = new plugin_set();
		(*_layers)[operationLayer] = set;
	}
	set->insert(c);
  return *c;
}

void asmg::WindowsPluginLibrarian::UnloadPluginLayer(PluginLayer layer, GlobalProviderCatalog& manager)
{
	layer_set::iterator it = _layers->find(layer);
	if (it == _layers->end())
	{	// no layer were found, ignore
		return;
	}
	plugin_set *set = it->second;
	// unload every plugin from the layer and remove from the global index
	plugin_set::iterator end = set->end();
	for (plugin_set::iterator it = set->begin(); it != end; ++it)
	{
		axis::String key = (*it)->GetFileName();
		(*it)->UnloadPlugin(manager);
		_plugins->erase(key);
		_loadStatus->erase(key);
	}
	// destroy layer set
	_layers->erase(layer);
	delete set;
}

void asmg::WindowsPluginLibrarian::LoadPluginLayer( PluginLayer layer, GlobalProviderCatalog& manager )
{
	layer_set::iterator it = _layers->find(layer);
	if (it == _layers->end())
	{	// there is no plugin on this layer to load; ignore
		return;
	}
	plugin_set *set = it->second;
	// register every plugin from the layer
	plugin_set::iterator end = set->end();
	for (plugin_set::iterator it = set->begin(); it != end; ++it)
	{
		bool loaded = (*_loadStatus)[(*it)->GetFileName()];
		if (!loaded)
		{
			(*it)->RegisterPlugin(manager);
			(*_loadStatus)[(*it)->GetFileName()] = true;
		}
	}
}

asmg::PluginConnector& asmg::WindowsPluginLibrarian::GetPluginConnector( 
  const axis::String& pluginFileName ) const
{	
	if (!ContainsPlugin(pluginFileName))
	{
		throw axis::foundation::ArgumentException();
	}
	axis::String s = pluginFileName;
	s.trim().to_lower_case();
	return *(*_plugins)[s];
}

bool asmg::WindowsPluginLibrarian::ContainsPlugin( const axis::String& pluginFileName ) const
{
	axis::String s = pluginFileName;
	s.trim().to_lower_case();
	return _plugins->find(s) != _plugins->end();
}

bool asmg::WindowsPluginLibrarian::IsPluginLoaded( const axis::String& pluginFileName ) const
{
	axis::String s = pluginFileName;
	s.trim().to_lower_case();
	load_map::iterator it = _loadStatus->find(s);
	if (it == _loadStatus->end())
	{
		throw axis::foundation::ArgumentException();
	}
	return it->second;
}

unsigned int asmg::WindowsPluginLibrarian::PluginCount( void ) const
{
	return (unsigned int)_plugins->size();
}

asmg::PluginLibrarian::Iterator asmg::WindowsPluginLibrarian::GetIterator( void ) const
{
	return Iterator(IteratorLogicImpl(_plugins->begin()), IteratorLogicImpl(_plugins->end()));
}

asmg::WindowsPluginLibrarian::IteratorLogicImpl::IteratorLogicImpl( const global_map::iterator& it )
{
	_it = it;
}

bool asmg::WindowsPluginLibrarian::IteratorLogicImpl::operator==( const IteratorLogic& logic ) const
{
	return _it == ((IteratorLogicImpl&)logic)._it;
}

bool asmg::WindowsPluginLibrarian::IteratorLogicImpl::operator!=( const IteratorLogic& logic ) const
{
	return !(*this == logic);
}

void asmg::WindowsPluginLibrarian::IteratorLogicImpl::Destroy( void ) const
{
	delete this;
}

asmg::PluginLibrarian::IteratorLogic& asmg::WindowsPluginLibrarian::IteratorLogicImpl::Clone( void ) const
{
	return *new IteratorLogicImpl(_it);
}

asmg::PluginLibrarian::IteratorLogic& asmg::WindowsPluginLibrarian::IteratorLogicImpl::GoNext( void )
{
	++_it;
	return *this;
}

asmg::PluginConnector& asmg::WindowsPluginLibrarian::IteratorLogicImpl::GetItem( void ) const
{
	return *_it->second;
}

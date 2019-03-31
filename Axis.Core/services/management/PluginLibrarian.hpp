#pragma once
#include "foundation/Axis.Core.hpp"
#include "AxisString.hpp"
#include "services/management/PluginLayer.hpp"
#include "services/management/PluginConnector.hpp"
#include "services/management/GlobalProviderCatalog.hpp"

namespace axis { namespace services { namespace management {

class AXISCORE_API PluginLibrarian
{
public:
	class AXISCORE_API IteratorLogic
	{
	public:
		virtual bool operator ==(const IteratorLogic& logic) const = 0;
		virtual bool operator !=(const IteratorLogic& logic) const = 0;
		virtual void Destroy(void) const = 0;
		virtual IteratorLogic& Clone(void) const = 0;
		virtual IteratorLogic& GoNext(void) = 0; 
		virtual PluginConnector& GetItem(void) const = 0;
	};
	class AXISCORE_API Iterator
	{
	public:
		Iterator(void);
		Iterator(const IteratorLogic& begin, const IteratorLogic& end);
		Iterator(const Iterator& it);
		~Iterator(void);
		Iterator& operator = (const Iterator& it);
		bool operator == (const Iterator& it) const;
		bool operator != (const Iterator& it) const;
		bool HasNext(void) const;
		Iterator& GoNext(void);
		PluginConnector& GetItem(void) const;
		PluginConnector *operator ->(void) const;
		PluginConnector& operator *(void) const;
	private:
		IteratorLogic *_current;
		IteratorLogic *_end;
	};

	virtual ~PluginLibrarian(void);
	virtual void Destroy(void) const = 0;
	virtual const axis::services::management::PluginConnector& AddConnector(
    const axis::String& pluginFileName, PluginLayer operationLayer) = 0;
	virtual void UnloadPluginLayer(PluginLayer layer, GlobalProviderCatalog& manager) = 0;
	virtual void LoadPluginLayer(PluginLayer layer, GlobalProviderCatalog& manager) = 0;
	virtual PluginConnector& GetPluginConnector(const axis::String& pluginFileName) const = 0;
	virtual bool ContainsPlugin(const axis::String& pluginFileName) const = 0;
	virtual bool IsPluginLoaded(const axis::String& pluginFileName) const = 0;
	virtual unsigned int PluginCount(void) const = 0;
	virtual Iterator GetIterator(void) const = 0;
};		

} } } // namespace axis::services::management

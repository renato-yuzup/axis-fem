#include "BootstrapAgent.hpp"

#include "services/management/GlobalProviderCatalogImpl.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "foundation/definitions/ConfigurationFileDescriptor.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aaa = axis::application::agents;
namespace af = axis::foundation;
namespace afd = af::definitions;
namespace asc = axis::services::configuration;
namespace asmg = axis::services::management;
namespace asmm = axis::services::messaging;

aaa::BootstrapAgent::BootstrapAgent( void ) :
_coreAgent(*new CorePluginAgent()), _extensibilityAgent(*new ExternalPluginAgent())
{
	_coreAgent.ConnectListener(*this);
	_extensibilityAgent.ConnectListener(*this);
	_manager = NULL;
	_configuration = NULL;
}

aaa::BootstrapAgent::~BootstrapAgent( void )
{
  delete &_extensibilityAgent;
  delete &_coreAgent;
	if (_manager != NULL) delete _manager;
	_manager = NULL;
}

void aaa::BootstrapAgent::SetUp( asc::ConfigurationScript& configuration )
{
	_configuration = &configuration;
}


void aaa::BootstrapAgent::Run( void )
{
	if (_configuration == NULL)
	{
		throw af::InvalidOperationException(
      _T("Cannot start bootstrap process without setting configuration data first."));
	}

	DispatchMessage(asmm::InfoMessage(0x100301, _T("Initiating bootstrap process...")));
	asmg::GlobalProviderCatalogImpl& manager = *new asmg::GlobalProviderCatalogImpl();

	_coreAgent.SetUp(manager);
	_extensibilityAgent.SetUp(*_configuration, manager);

	// load core providers
	DispatchMessage(asmm::InfoMessage(0x100302, _T("PHASE 0: Loading core providers...")));
	manager.SetRunMode(asmg::GlobalProviderCatalogImpl::kPhase0);
	_coreAgent.RegisterCoreProviders();

	// load system customizable providers
	manager.SetRunMode(asmg::GlobalProviderCatalogImpl::kPhase0dot5);
	_coreAgent.RegisterSystemCustomizableProviders();

	// Now we need to load providers from external plugins. First, 
	// see if we have a plugins section declared in the
	// configuration script.
	if (DefinesPlugins())
	{	
		asc::ConfigurationScript& pluginsConfig = 
      _configuration->GetSection(afd::ConfigurationFileDescriptor::PluginsSectionName);

		_extensibilityAgent.SetUp(pluginsConfig, manager);

		// load system customization plugins
		DispatchMessage(asmm::InfoMessage(0x100302, 
      _T("PHASE 1: Loading external system customization plugins...")));
		manager.SetRunMode(asmg::GlobalProviderCatalogImpl::kPhase1);
		_extensibilityAgent.LoadSystemCustomizerPlugins();

		// load non-volatile plugins
		DispatchMessage(asmm::InfoMessage(0x100302, 
      _T("PHASE 2: Loading external non-volatile plugins...")));
		manager.SetRunMode(asmg::GlobalProviderCatalogImpl::kPhase2);
		_extensibilityAgent.LoadNonVolatilePlugins();

		DispatchMessage(asmm::InfoMessage(0x100302, 
      _T("PHASE 3: Loading external volatile plugins...")));
		manager.SetRunMode(asmg::GlobalProviderCatalogImpl::kPhase3);
		_extensibilityAgent.LoadVolatilePlugins();

		DispatchMessage(asmm::InfoMessage(0x100303, String(_T("Plugins loaded: ")) + 
      String::int_parse(_extensibilityAgent.GetPluginCount())));
	}
	else
	{	// section doesn't exist, so we have nothing to do :)
		DispatchMessage(asmm::InfoMessage(0x100301, 
      _T("No plugin configuration section was found in the settings file.")));
		DispatchMessage(asmm::InfoMessage(0x100301, 
      _T("Phase 1 and subsequent phases were skipped because plugin configuration settings is missing.")));
		DispatchMessage(asmm::InfoMessage(0x100303, String(_T("Plugins loaded: 0"))));
	}
	_extensibilityAgent.EnumerateLoadedPlugins();	
	DispatchMessage(asmm::InfoMessage(0x100302, _T("Bootstrap process complete.")));
	_manager = &manager;
}

asmg::GlobalProviderCatalog& aaa::BootstrapAgent::GetModuleManager( void )
{
	return *_manager;
}

const asmg::GlobalProviderCatalog& aaa::BootstrapAgent::GetModuleManager( void ) const
{
	return *_manager;
}

bool aaa::BootstrapAgent::DefinesPlugins( void ) const
{
	return _configuration->ContainsSection(afd::ConfigurationFileDescriptor::PluginsSectionName);
}

asmg::PluginLink aaa::BootstrapAgent::GetPluginLinkInfo( size_type index ) const
{
	return _extensibilityAgent.GetPluginLinkInfo(index);
}

size_type aaa::BootstrapAgent::GetPluginLinkCount( void ) const
{
	return _extensibilityAgent.GetPluginCount();
}

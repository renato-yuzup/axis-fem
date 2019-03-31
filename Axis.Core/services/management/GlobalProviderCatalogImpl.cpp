#include "GlobalProviderCatalogImpl.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "foundation/AxisException.hpp"
#include "AxisString.hpp"
#include "log_messages/manager_messages.hpp"
#include "services/management/Provider.hpp"
#include <list>
#include "SystemModuleProxy.hpp"
#include "UserModuleProxy.hpp"
#include "foundation/PermissionDeniedException.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/WarningMessage.hpp"

namespace af = axis::foundation;
namespace asmm = axis::services::messaging;
namespace asmg = axis::services::management;

asmg::GlobalProviderCatalogImpl::GlobalProviderCatalogImpl( void ) :
	_features(*new feature_tree())
{
	_runMode = kPhase0;
}

asmg::GlobalProviderCatalogImpl::~GlobalProviderCatalogImpl( void )
{
	UnloadModules();
	// delete module tree
	delete &_features;
}

void asmg::GlobalProviderCatalogImpl::UnloadModules( void )
{
	typedef feature_tree::value_type::second_type item_type;
	std::list<item_type> nodes;
	// copy item nodes to another container
	for (feature_tree::iterator it = _features.begin(); it != _features.end(); ++it)
	{
		nodes.push_back(it->second);
	}
	// unregister and delete node by node
	while(nodes.size() > 0)
	{
		// unregister provider
		ProviderProxy *proxy = nodes.front();
		DoUnregisterProvider(proxy->GetProvider());
		nodes.pop_front();
	}
}

void asmg::GlobalProviderCatalogImpl::RegisterProvider( Provider& provider )
{
	// check if we are replacing an existing provider
	if (ExistsProvider(provider.GetFeaturePath()))
	{	// yes, we are
		ProviderProxy& proxy = GetProviderProxy(provider.GetFeaturePath());
		// ignore we are re-registering a provider
		if (&proxy.GetProvider() == &provider) return;
		// check security attributes if we can do it
		if (!IsModifiable(proxy))
		{	// security check failed
			Provider& existentProvider = proxy.GetProvider();
			String actualProviderName; 
      StringEncoding::AssignFromASCII(provider.GetFeatureName(), actualProviderName);
			String existentProviderName; 
      StringEncoding::AssignFromASCII(existentProvider.GetFeatureName(), existentProviderName);
			String providerPath; 
      StringEncoding::AssignFromASCII(provider.GetFeaturePath(), providerPath);
			String s = AXIS_LOG_MANAGER_REGISTRATIONDSECURITY_ERROR;
			s = s.replace(_T("%1"), actualProviderName)
           .replace(_T("%2"), existentProviderName)
           .replace(_T("%3"), providerPath);
			DispatchMessage(asmm::ErrorMessage(0x300101, s, _T("Plugin registration error"), 
                      asmm::ErrorMessage::ErrorCritical));

			throw axis::foundation::PermissionDeniedException(AXIS_LOG_MANAGER_REGISTRATIONDSECURITY_ERROR);
		}
		else
		{	// security check ok, detach current provider for later replacement
			UnregisterProvider(proxy.GetProvider());
		}
	}

	// add new provider
	_features[provider.GetFeaturePath()] = &CreateProxy(provider);

	// try to execute post processing
	try
	{
		provider.PostProcessRegistration(*this);
	}
	catch (axis::foundation::AxisException& e)
	{	// registration failed with an known error
		String actualProviderName; 
    StringEncoding::AssignFromASCII(provider.GetFeatureName(), actualProviderName);
		String s = AXIS_LOG_MANAGER_POSTREGISTRATION_ERROR;
		s = s.replace(_T("%1"), actualProviderName) + _T("\n");
		DispatchMessage(asmm::ErrorMessage(0x300102, s, _T("Plugin registration error"), 
                    e, asmm::ErrorMessage::ErrorCritical));

		// roll back registration
		_features.erase(provider.GetFeaturePath());

		throw axis::foundation::InvalidOperationException(&e);
	}
	catch (...)
	{	// failed with some unknown error
		String actualProviderName; 
    StringEncoding::AssignFromASCII(provider.GetFeatureName(), actualProviderName);
		String s = AXIS_LOG_MANAGER_POSTREGISTRATION_ERROR;
		s = s.replace(_T("%1"), actualProviderName) + _T("\n");
		DispatchMessage(asmm::ErrorMessage(0x300102, s, _T("Plugin registration error"), 
                    asmm::ErrorMessage::ErrorCritical));
		// roll back registration
		_features.erase(provider.GetFeaturePath());
		throw axis::foundation::InvalidOperationException();
	}
}

void asmg::GlobalProviderCatalogImpl::UnregisterProvider( Provider& provider )
{
	if (!ExistsProvider(provider.GetFeaturePath()))
	{
		throw axis::foundation::InvalidOperationException();
	}
	// check if this provider can be modified
	ProviderProxy& proxy = *_features[provider.GetFeaturePath()];
	if (!IsModifiable(proxy))
	{	// cannot remove this provider, insufficient privileges
		throw axis::foundation::PermissionDeniedException();
	}
	DoUnregisterProvider(provider);
}

asmg::Provider& asmg::GlobalProviderCatalogImpl::GetProvider( const char *providerPath ) const
{
	return GetProviderProxy(providerPath).GetProvider();
}

bool asmg::GlobalProviderCatalogImpl::ExistsProvider( const char *providerPath ) const
{
	// let's try to get the item
	try
	{
		GetProviderProxy(providerPath);
		return true;
	}
	catch (axis::foundation::ElementNotFoundException&)
	{	// failed, item doesn't exist
		return false;
	}
}

void asmg::GlobalProviderCatalogImpl::RegisterProviderWithoutProxy( ProviderProxy& proxy )
{
	Provider& provider = proxy.GetProvider();
	// check if we are replacing an existing proxy
	if (ExistsProvider(provider.GetFeaturePath()))
	{	// yes, we are; fail
		Provider& existentProvider = GetProvider(provider.GetFeaturePath());
		String actualProviderName; 
    StringEncoding::AssignFromASCII(provider.GetFeatureName(), actualProviderName);
		String existentProviderName; 
    StringEncoding::AssignFromASCII(existentProvider.GetFeatureName(), existentProviderName);
		String providerPath; 
    StringEncoding::AssignFromASCII(provider.GetFeaturePath(), providerPath);

		String s = AXIS_LOG_MANAGER_REGISTRATIONDSECURITY_ERROR;
		s = s.replace(_T("%1"), actualProviderName).replace(_T("%2"), existentProviderName)
         .replace(_T("%3"), providerPath);
		DispatchMessage(asmm::ErrorMessage(0x300101, s, _T("Plugin registration error"), 
                    asmm::ErrorMessage::ErrorCritical));
		throw axis::foundation::InvalidOperationException(AXIS_LOG_MANAGER_REGISTRATIONDSECURITY_ERROR);
	}
	// add new proxy
	_features[provider.GetFeaturePath()] = &proxy;
	// try to execute post processing
	try
	{
		provider.PostProcessRegistration(*this);
	}
	catch (axis::foundation::AxisException& e)
	{	// registration failed with an known error
		String actualProviderName; 
    StringEncoding::AssignFromASCII(provider.GetFeatureName(), actualProviderName);
		String s = AXIS_LOG_MANAGER_POSTREGISTRATION_ERROR;
		s = s.replace(_T("%1"), actualProviderName) + _T("\n");
		DispatchMessage(asmm::ErrorMessage(0x300102, s, _T("Plugin registration error"), e, 
                    asmm::ErrorMessage::ErrorCritical));
		// roll back registration
		_features.erase(provider.GetFeaturePath());
		throw axis::foundation::InvalidOperationException(&e);
	}
	catch (...)
	{	// failed with some unknown error
		String actualProviderName; 
    StringEncoding::AssignFromASCII(provider.GetFeatureName(), actualProviderName);
		String s = AXIS_LOG_MANAGER_POSTREGISTRATION_ERROR;
		s = s.replace(_T("%1"), actualProviderName) + _T("\n");
		DispatchMessage(asmm::ErrorMessage(0x300102, s, _T("Plugin registration error"), 
                    asmm::ErrorMessage::ErrorCritical));
		// roll back registration
		_features.erase(provider.GetFeaturePath());
		throw axis::foundation::InvalidOperationException();
	}
}

void asmg::GlobalProviderCatalogImpl::SetRunMode( RunMode mode )
{
	_runMode = mode;
}

asmg::ProviderProxy& asmg::GlobalProviderCatalogImpl::GetProviderProxy( const char *providerPath ) const
{
	std::string s(providerPath);
	feature_tree::iterator it = _features.find(s);
	if (it == _features.end())
	{	// failed
		throw axis::foundation::ElementNotFoundException();
	}
	return *it->second;
}

bool asmg::GlobalProviderCatalogImpl::IsModifiable( const ProviderProxy& proxyToOverwrite ) const
{
	switch (_runMode)
	{
	case kPhase0:	// (only built-in modules are loaded here)
		// in bootstrap mode we cannot overwrite any module; if we do, it might
		// represent an internal error
		return false;
	case kPhase0dot5:	// (overridable system modules are loaded here)
		// in this "semi-mode", overridable system providers are loaded here
		return false;
	case kPhase1:	// (custom system modules are loaded here)
		// in system customization mode, it is possible to override the system modules
		// loaded at the previous phase, but it is not possible to overwrite
		// modules loaded at the same phase
		return !proxyToOverwrite.IsBootstrapModule() && proxyToOverwrite.IsSystemModule() && 
           !proxyToOverwrite.IsCustomSystemModule();
	case kPhase2:	// (non-volatile user modules are loaded here)
		// in system extensibility mode, user-mode modules are loaded as non-volatile,
		// that is, it can't overwrite any module loaded at a previous phase but cannot be
		// overwritten at a further phase; also, modules loaded at this phase cannot
		// overwrite each other
		return false;	// in other words, overwrite is not allowed in this context
	case kPhase3:		// (only user modules are loaded here)
		// in user platform build mode, user modules are loaded; these modules
		// cannot overwrite any previously loaded module, but can overwrite
		// functionalities of others loaded at the same phase
		return proxyToOverwrite.IsUserModule() && !proxyToOverwrite.IsNonVolatileUserModule();
	}

	// just to shut up the compiler warnings
	return false;
}

asmg::ProviderProxy& asmg::GlobalProviderCatalogImpl::CreateProxy( Provider& provider ) const
{
	switch (_runMode)
	{
	case kPhase0:
		return *new SystemModuleProxy(provider, true, false);
	case kPhase0dot5:
		return *new SystemModuleProxy(provider, false, false);
	case kPhase1:
		return *new SystemModuleProxy(provider, false, true);
	case kPhase2:
		return *new UserModuleProxy(provider, true);
	case kPhase3:
		return *new UserModuleProxy(provider, false);
	}
	// unexpected state (also to shut the compiler warnings :) )
	throw axis::foundation::InvalidOperationException();
}

void asmg::GlobalProviderCatalogImpl::DoUnregisterProvider( Provider& provider )
{
	// remove from provider list
	_features.erase(provider.GetFeaturePath());
	// tell feature to unload
	try
	{
		provider.UnloadModule(*this);
	}
	catch (axis::foundation::AxisException& e)
	{	// registration failed with a known error; log and continue
		String providerName; 
    StringEncoding::AssignFromASCII(provider.GetFeatureName(), providerName);
		String providerPath; 
    StringEncoding::AssignFromASCII(provider.GetFeaturePath(), providerPath);
		String s = AXIS_LOG_MANAGER_UNLOAD_ERROR;
		s = s.replace(_T("%1"), providerName).replace(_T("%2"), providerPath);
		DispatchMessage(asmm::WarningMessage(0x200101, s, _T("Plugin failed to unload"), e, 
                    asmm::WarningMessage::WarningHigh));
	}
	catch (...)
	{	// failed with some unknown error
		String providerName; 
    StringEncoding::AssignFromASCII(provider.GetFeatureName(), providerName);
		String providerPath; 
    StringEncoding::AssignFromASCII(provider.GetFeaturePath(), providerPath);
		String s = AXIS_LOG_MANAGER_UNLOAD_ERROR;
		s = s.replace(_T("%1"), providerName).replace(_T("%2"), providerPath);
		DispatchMessage(asmm::WarningMessage(0x200101, s, _T("Plugin failed to unload"), 
                    asmm::WarningMessage::WarningHigh));
	}
}
/* Member of KeyLessThanComparator */
bool asmg::GlobalProviderCatalogImpl::KeyLessThanComparator::operator()( 
  const provider_key_type& key1, const provider_key_type& key2 ) const
{
	return key1.compare(key2) < 0;
}

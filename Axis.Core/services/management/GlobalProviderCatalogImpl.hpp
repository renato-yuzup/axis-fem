/// <summary>
/// Contains definition for the class axis::services::management::GlobalProviderCatalogImpl.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "services/management/GlobalProviderCatalog.hpp"
#include "boost/property_tree/ptree.hpp"
#include <string>
#include "ProviderProxy.hpp"
#include <map>

namespace axis { namespace services { namespace management {

/// <summary>
/// Implements a management interface which centers requests for features.
/// </summary>
class GlobalProviderCatalogImpl : public GlobalProviderCatalog
{
public:
	enum RunMode
	{
		kPhase0		= 0,
		kPhase0dot5	= 1,
		kPhase1		= 2,
		kPhase2		= 3,
		kPhase3		= 4
	};

  /// <summary>
	/// Creates a new instance of this class.
	/// </summary>
	GlobalProviderCatalogImpl(void);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	~GlobalProviderCatalogImpl(void);

	/// <summary>
	/// Register a new feature provider.
	/// </summary>
	/// <param name="provider">The feature provider to be registered.</param>
	virtual void RegisterProvider(Provider& provider);

	/// <summary>
	/// Register a new feature provider not caring about object overwrite security during
	/// startup module registration modes.
	/// </summary>
	/// <remarks>
	/// This method is called only internally to append root block providers to the manager
	///  and should not be used elsewhere.
	/// </remarks>
	/// <param name="provider">The proxy provider to be registered.</param>
	void RegisterProviderWithoutProxy(ProviderProxy& proxy);

	/// <summary>
	/// Sets the run mode which determines providers security attributes to be passed to
	/// newly registered providers.
	/// </summary>
	void SetRunMode(RunMode mode);

	/// <summary>
	/// Removes a feature provider.
	/// </summary>
	/// <param name="provider">The feature provider to be registered.</param>
	virtual void UnregisterProvider(Provider& provider);

	/// <summary>
	/// Returns the feature provider with the specified fully qualified feature name.
	/// </summary>
	virtual Provider& GetProvider(const char *providerPath) const;

	/// <summary>
	/// Returns if the specified provider is registered with this object.
	/// </summary>
	/// <param name="providerPath">Fully qualified feature name of the provider.</param>
	virtual bool ExistsProvider(const char *providerPath) const;
private:
  typedef std::string provider_key_type;
  class KeyLessThanComparator
  {
  public:
    bool operator()(const provider_key_type& key1, const provider_key_type& key2) const;
  };
  typedef std::map<provider_key_type, ProviderProxy *, KeyLessThanComparator> feature_tree;

  ProviderProxy& CreateProxy(Provider& provider) const;
  ProviderProxy& GetProviderProxy(const char *providerPath) const;
  void UnloadModules(void);
  void DoUnregisterProvider(Provider& provider);

  bool IsModifiable(const ProviderProxy& proxyToOverwrite) const;

  feature_tree& _features;		// list of features registered
  RunMode _runMode;
};

} } } // namespace axis::services::management

/// <summary>
/// Contains definition for the abstract class axis::services::management::GlobalProviderCatalog.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/CollectorEndpoint.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			class AXISSYSTEMBASE_API Provider;

			/// <summary>
			/// Represents a coordinator from which feature providers can be
			/// registered or obtained. </summary>
			class AXISSYSTEMBASE_API GlobalProviderCatalog  : public axis::services::messaging::CollectorEndpoint
			{
			public:
				virtual ~GlobalProviderCatalog(void);

				/// <summary>
				/// Register a new feature provider.
				/// </summary>
				/// <param name="provider">The feature provider to be registered.</param>
				virtual void RegisterProvider(Provider& provider) = 0;

				/// <summary>
				/// Removes a feature provider.
				/// </summary>
				/// <param name="provider">The feature provider to be registered.</param>
				virtual void UnregisterProvider(Provider& provider) = 0;

				/// <summary>
				/// Returns the feature provider with the specified fully qualified feature name.
				/// </summary>
				virtual Provider& GetProvider(const char *providerPath) const = 0;

				/// <summary>
				/// Returns if the specified provider is registered with this object.
				/// </summary>
				/// <param name="providerPath">Fully qualified feature name of the provider.</param>
				virtual bool ExistsProvider(const char *providerPath) const = 0;
			};
		}
	}
}
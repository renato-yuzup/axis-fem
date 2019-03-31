#include "Provider.hpp"

axis::services::management::Provider::~Provider(void)
{
	// nothing to do here
}

void axis::services::management::Provider::PostProcessRegistration( axis::services::management::GlobalProviderCatalog& manager )
{
	/* Nothing to do at base implementation */
}

void axis::services::management::Provider::UnloadModule( axis::services::management::GlobalProviderCatalog& manager )
{
	/* Nothing to do at base implementation */
}
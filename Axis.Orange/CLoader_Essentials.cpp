#include "CLoader_Essentials.hpp"
#include "services/management/EssentialsPluginLoader.hpp"

AXISORANGE_API axis::services::management::PluginLoader *AxisPluginLoader_GetLoader(void)
{
	// create plugin startup object
	return new axis::services::management::EssentialsPluginLoader();
}

#include "CLoader_MyMaterials.hpp"
#include "services/management/MyMaterialsPluginLoader.hpp"

AXISSTANDARDMATERIALS_API axis::services::management::PluginLoader *AxisPluginLoader_GetLoader(void)
{
	// create plugin startup object
	return new axis::services::management::MyMaterialsPluginLoader();
}
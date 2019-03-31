#include "CLoader_MyElements.hpp"
#include "services/management/MyElementsPluginLoader.hpp"

AXISSTANDARDELEMENTS_API axis::services::management::PluginLoader *AxisPluginLoader_GetLoader(void)
{
	// create plugin startup object
	return new axis::services::management::MyElementsPluginLoader();
}
#include "CLoader_Solver.hpp"
#include "services/management/MySolverPluginLoader.hpp"

AXISSOLVER_API axis::services::management::PluginLoader *AxisPluginLoader_GetLoader(void)
{
	// create plugin startup object
	return new axis::services::management::MySolverPluginLoader();
}

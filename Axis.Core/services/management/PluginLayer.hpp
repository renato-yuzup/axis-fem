#pragma once
#include "foundation/Axis.Core.hpp"

namespace axis { namespace services { namespace management {

enum AXISCORE_API PluginLayer
{
	kPhase0_SystemPhase,
	kPhase1_SystemCustomizationPhase,
	kPhase2_NonVolatilePhase,
	kPhase3_VolatilePhase
};

} } } // namespace axis::services::management

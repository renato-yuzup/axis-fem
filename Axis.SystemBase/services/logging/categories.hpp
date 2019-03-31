#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis
{
	namespace services
	{
		namespace logging
		{
			enum AXISSYSTEMBASE_API Severity
			{
				SeverityLow = 0,
				SeverityNormal = 500,
				SeverityHigh = 1000,
				SeverityCritical = 5000
			};

			enum AXISSYSTEMBASE_API InfoLevel
			{
				InfoDebugMode = 0,
				InfoVerbose = 100,
				InfoSupplementalInfo = 500,
				InfoNormal = 5000
			};
		}
	}
}


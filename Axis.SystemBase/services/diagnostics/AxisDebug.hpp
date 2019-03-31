#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis
{
	namespace services
	{
		namespace diagnostics
		{
			class AXISSYSTEMBASE_API AxisDebug
			{
			private:
				AxisDebug(void);
			public:
				~AxisDebug(void);

				static void MarkAssertionFailed(const char * expr, const char * function, const char * file, long line);

			};
		}
	}
}

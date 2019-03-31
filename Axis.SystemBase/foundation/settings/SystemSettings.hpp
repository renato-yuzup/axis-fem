#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace foundation
	{
		namespace settings
		{
			class AXISSYSTEMBASE_API SystemSettings
			{
			private:
				SystemSettings(void);
			public:
				static const axis::String::char_type * DefaultConfigurationFile;
				static const axis::String::char_type * LocaleFolderName;
				static const axis::String::char_type * NewLine;
			};	
		}
	}
}
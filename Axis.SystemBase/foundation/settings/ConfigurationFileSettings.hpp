#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace foundation
	{
		namespace settings
		{
			class AXISSYSTEMBASE_API ConfigurationFileSettings
			{
			private:
				ConfigurationFileSettings(void);
			public:
				static const axis::String::char_type * ConfigurationRoot;
				static const axis::String::char_type * DirectoriesSection;
				static const axis::String::char_type * DirectoriesElements;
				static const axis::String::char_type * DirectoriesElementsFeatureAttr;
				static const axis::String::char_type * DirectoriesElementsLocationAttr;
			};		
		}
	}
}
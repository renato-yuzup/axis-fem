#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "ConfigurationScript.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace configuration
		{
			class AXISCOMMONLIBRARY_API ScriptFactory
			{
			private:
				ScriptFactory(void);
			public:
				~ScriptFactory(void);

				static ConfigurationScript& ReadFromXmlFile(const axis::String& fileName);
			};	
		}
	}

}


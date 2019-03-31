#pragma once
#include "AxisString.hpp"

namespace axis
{
	namespace Interop
	{	
		namespace foundation
		{
			ref class StringWrapService
			{
			private:
				StringWrapService(void);
			public:
				static axis::String WrapToAxisString(System::String ^str);
				static System::String ^UnwrapFromAxisString(const axis::String& str);
			};		
		}
	}
}


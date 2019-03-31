#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace foundation
	{
		namespace assertion
		{
			/**********************************************************************************************//**
			 * @class	Ascertainable
			 *
			 * @brief	Represents any object which can have its type checked.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	26 jun 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Ascertainable
			{
			public:
				virtual ~Ascertainable(void);

				/**********************************************************************************************//**
				 * @brief	Gets the type name identifier of this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	26 jun 2012
				 *
				 * @return	The type name.
				 **************************************************************************************************/
				virtual axis::String GetTypeName(void) const = 0;
			};
		
		}
	}
}


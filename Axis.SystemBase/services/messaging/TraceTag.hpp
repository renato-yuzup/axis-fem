#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Represents a tag object attached to a trace information
			 * 			of a message.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	25 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API TraceTag
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~TraceTag(void);

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Makes a deep copy of this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A reference to the newly created object.
				 **************************************************************************************************/
				virtual TraceTag& Clone(void) const = 0;
			};
		}
	}
}


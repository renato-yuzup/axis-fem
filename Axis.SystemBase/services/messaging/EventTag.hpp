#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/collections/Collectible.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Represents any tag attached to an event message.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	axis::foundation::collections::Collectible
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API EventTag : public axis::foundation::collections::Collectible
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~EventTag(void);

				/**********************************************************************************************//**
				 * @brief	Destroys this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual void Destroy(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Makes a deep copy of this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	A copy of this object.
				 **************************************************************************************************/
				virtual EventTag& Clone(void) const = 0;
			};
		}
	}
}


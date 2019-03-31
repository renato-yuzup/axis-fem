#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/Assertion/Ascertainable.hpp"
#include "foundation/collections/Collectible.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			namespace metadata
			{
				/**********************************************************************************************//**
				 * @brief	Represents a metadatum object which contains
				 * 			addition informations for a message object. These
				 * 			informations normally are not closely related to the
				 * 			message subject, but might be usable for diagnostics
				 * 			purposes, for example.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @sa	axis::foundation::assertion::Ascertainable
				 * @sa	axis::foundation::collections::Collectible
				 * @sa	axis::services::messaging::Message
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API Metadatum : public axis::foundation::assertion::Ascertainable, public axis::foundation::collections::Collectible
				{
				public:

					/**********************************************************************************************//**
					 * @brief	Default constructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					Metadatum(void);

					/**********************************************************************************************//**
					 * @brief	Destructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					virtual ~Metadatum(void);

					/**********************************************************************************************//**
					 * @brief	Makes a deep copy of this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A copy of this object.
					 **************************************************************************************************/
					virtual Metadatum& Clone(void) const = 0;

					/**********************************************************************************************//**
					 * @brief	Destroys this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					virtual void Destroy(void) const = 0;

					/**********************************************************************************************//**
					 * @brief	Gets the type name for the class of this object.
					 * 			@remark This string will be used to identify instances of
					 * 					this class in metadata collection.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	The type name.
					 **************************************************************************************************/
					virtual axis::String GetTypeName( void ) const = 0;
				};
			
			}
		}
	}
}


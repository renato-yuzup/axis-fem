#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/Message.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Declares a message which has been dispatched as a side
			 * 			effect of reaching a certain point in current analysis
			 * 			computation.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	Message
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API ResultMessage : public Message
			{
			private:
				axis::String _description;
			protected:
				/**********************************************************************************************//**
				 * @brief	Clears resources used by this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual void DoDestroy( void ) const = 0;

				/**********************************************************************************************//**
				 * @brief	Executes the clone operation.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 *
				 * @return	A reference to the new object.
				 **************************************************************************************************/
				virtual Message& DoClone( id_type id ) const = 0;
			public:
				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 **************************************************************************************************/
				ResultMessage(Message::id_type id);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id		   	The message identifier.
				 * @param	description	A description string carried by the message.
				 **************************************************************************************************/
				ResultMessage(Message::id_type id, const axis::String& description);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~ResultMessage(void);

				/**********************************************************************************************//**
				 * @brief	Returns the description string for this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	A string containing the description.
				 **************************************************************************************************/
				axis::String GetDescription(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an event message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an event message, false otherwise.
				 **************************************************************************************************/
				virtual bool IsEvent( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a result message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a result message, false otherwise.
				 **************************************************************************************************/
				virtual bool IsResult( void ) const;
			};
		}
	}
}


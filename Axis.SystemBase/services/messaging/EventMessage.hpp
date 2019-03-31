#pragma once
#include "AxisString.hpp"
#include "foundation/collections/ObjectMap.hpp"
#include "services/messaging/Message.hpp"
#include "services/messaging/EventTag.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Declares a message which has been dispatched to notify
			 * 			listeners about a specific state of an object or an
			 * 			entire system of the program.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	Message
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API EventMessage : public Message
			{
			private:
				axis::String _title;
				axis::String _description;

				axis::foundation::collections::ObjectMap& _tags;

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
				virtual Message& DoClone( id_type id ) const;

				/**********************************************************************************************//**
				 * @brief	Creates a copy this object and its specific properties.
				 * 			@remark	This method is called by the method Clone() of
				 * 					base class Message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 *
				 * @return	A reference to new object.
				 **************************************************************************************************/
				virtual Message& CloneMyself(id_type id) const = 0;
			public:
				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 **************************************************************************************************/
				EventMessage(Message::id_type id);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 **************************************************************************************************/
				EventMessage(Message::id_type id, const axis::String& message);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	title  	The message title.
				 **************************************************************************************************/
				EventMessage(Message::id_type id, const axis::String& message, const axis::String& title);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~EventMessage(void);

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

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an error message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an error message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsError(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a warning message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a warning message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsWarning(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an informational message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an informational message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsInfo(void) const = 0;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a log message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a log message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsLogEntry(void) const = 0;

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
				 * @brief	Returns the title string for this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	A string containing the title.
				 **************************************************************************************************/
				axis::String GetTitle(void) const;

				/**********************************************************************************************//**
				 * @brief	Appends a tag object to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	tagName	   	Exclusive identifier of the tag object.
				 * @param [in,out]	tag	The tag object.
				 **************************************************************************************************/
				void AppendTag(const axis::String& tagName, EventTag& tag);

				/**********************************************************************************************//**
				 * @brief	Queries if a tag object is attached to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	tagName	Tag identifier.
				 *
				 * @return	true if the tag is attached, false otherwise.
				 **************************************************************************************************/
				bool ContainsTag(const axis::String& tagName) const;

				/**********************************************************************************************//**
				 * @brief	Removes a tag object from this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	tagName	Tag identifier.
				 **************************************************************************************************/
				void EraseTag(const axis::String& tagName);

				/**********************************************************************************************//**
				 * @brief	Removes all attached tag objects from this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				void ClearAllTags(void);

				/**********************************************************************************************//**
				 * @brief	Returns the number of tag objects attached to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	A non-negative number representing the number 
				 * 			of attached tags.
				 **************************************************************************************************/
				size_type TagCount(void) const;
			};
		}
	}
}


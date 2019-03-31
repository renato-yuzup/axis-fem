#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/EventMessage.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Declares an event message which carries only 
			 * 			informational contents about a specific event.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	EventMessage
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API InfoMessage : public EventMessage
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Values that represent informational level carried by a message.
				 **************************************************************************************************/
				enum InfoLevel
				{
					///< Any relevant information.
					InfoNormal = 8,
					///< Any operational information to be written in a log output.
					InfoLogLevel = 4,
					///< Any operational information.
					InfoVerbose = 2,
					///< Any operational information plus informations relevant for debugging purposes.
					InfoDebugLevel = 1
				};
			private:
				InfoLevel _level;
			public:
				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 **************************************************************************************************/
				InfoMessage(Message::id_type id);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id   	The message identifier.
				 * @param	level	The informational level of the message.
				 **************************************************************************************************/
				InfoMessage(Message::id_type id, InfoLevel level);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 **************************************************************************************************/
				InfoMessage(Message::id_type id, const axis::String& message);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	level	The informational level of the message.
				 **************************************************************************************************/
				InfoMessage(Message::id_type id, const axis::String& message, InfoLevel level);

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
				InfoMessage(Message::id_type id, const axis::String& message, const axis::String& title);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	title  	The message title.
				 * @param	level	The informational level of the message.
				 **************************************************************************************************/
				InfoMessage(Message::id_type id, const axis::String& message, const axis::String& title, InfoLevel level);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~InfoMessage(void);

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an error message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an error message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsError( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a warning message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a warning message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsWarning( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is an informational message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is an informational message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsInfo( void ) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this object is a log message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if it is a log message, false if otherwise.
				 **************************************************************************************************/
				virtual bool IsLogEntry( void ) const;

				/**********************************************************************************************//**
				 * @brief	Returns the informational level of this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The severity.
				 **************************************************************************************************/
				InfoLevel GetInfoLevel(void) const;

			protected:
				/**********************************************************************************************//**
				 * @brief	Clears resources used by this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual void DoDestroy( void ) const;

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
				virtual Message& CloneMyself( id_type id ) const;
			};
		}
	}
}


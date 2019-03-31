#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/AxisException.hpp"
#include "services/messaging/EventMessage.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Declares an event message triggered by an error condition
			 * 			in an object.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	EventMessage
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API ErrorMessage : public EventMessage
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Values that represent error severity.
				 **************************************************************************************************/
				enum Severity
				{
					///< An error with minor consequences, normally with a possible workaround.
					ErrorLow = 1,

					///< An error which can cause termination of the current process.
					ErrorNormal = 2,

					///< An error which normally has been triggered by an external source 
					// and has serious consequences in the program execution.
					ErrorCritical = 4
				};
			private:
				Severity _level;
				const axis::foundation::AxisException *_exception;
			public:

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id   	The message identifier.
				 * @param	level	The severity level of the error.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, Severity level);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::String& message);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	level  	The severity level of the error.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::String& message, Severity level);

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
				ErrorMessage(Message::id_type id, const axis::String& message, const axis::String& title);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	title  	The message title.
				 * @param	level  	The severity level of the error.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::String& message, const axis::String& title, Severity level);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	The message identifier.
				 * @param	e 	The exception which triggered this message.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::foundation::AxisException& e);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id   	The message identifier.
				 * @param	e 		The exception which triggered this message.
				 * @param	level  	The severity level of the error.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::foundation::AxisException& e, Severity level);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	e 		The exception which triggered this message.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::String& message, const axis::foundation::AxisException& e);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	e 		The exception which triggered this message.
				 * @param	level  	The severity level of the error.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::String& message, const axis::foundation::AxisException& e, Severity level);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	title  	The message title.
				 * @param	e 		The exception which triggered this message.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::String& message, const axis::String& title, const axis::foundation::AxisException& e);

				/**********************************************************************************************//**
				 * @brief	Creates a new message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	id	   	The message identifier.
				 * @param	message	A description string carried by the message.
				 * @param	title  	The message title.
				 * @param	e 		The exception which triggered this message.
				 * @param	level  	The severity level of the error.
				 **************************************************************************************************/
				ErrorMessage(Message::id_type id, const axis::String& message, const axis::String& title, const axis::foundation::AxisException& e, Severity level);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~ErrorMessage(void);

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
				 * @brief	Returns the severity level of this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The severity.
				 **************************************************************************************************/
				Severity GetSeverity(void) const;

				/**********************************************************************************************//**
				 * @brief	Queries if this message was associated to an exception.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	true if an associated exception exists, false otherwise.
				 **************************************************************************************************/
				bool HasAssociatedException(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the associated exception to this message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	A reference to the associated exception.
				 **************************************************************************************************/
				const axis::foundation::AxisException& GetAssociatedException(void) const;

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

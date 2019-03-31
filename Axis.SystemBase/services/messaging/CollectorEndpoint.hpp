#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/MessageListener.hpp"
#include "foundation/collections/ObjectSet.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Represents any object which is capable of dispatching messages
			 * 			to a limited number of listener objects. 
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API CollectorEndpoint
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Defines an alias representing the iterator for this object.
				 **************************************************************************************************/
				typedef axis::foundation::collections::ObjectSet::Iterator Iterator;
			private:
				axis::foundation::collections::ObjectSet& _eventListeners;

        virtual void AddTracingInformation(Message& message) const;
			public:

				/**********************************************************************************************//**
				 * @brief	Default constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				CollectorEndpoint(void);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~CollectorEndpoint(void);

				/**********************************************************************************************//**
				 * @brief	Connects a listener which will receive messages
				 * 			of this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param [in,out]	listener	The listener.
				 **************************************************************************************************/
				void ConnectListener(MessageListener& listener);

				/**********************************************************************************************//**
				 * @brief	Disconnects a listener so that it no longer receives
				 * 			messages directly from this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param [in,out]	listener	The listener.
				 **************************************************************************************************/
				void DisconnectListener(MessageListener& listener);

				/**********************************************************************************************//**
				 * @brief	Disconnects all listeners so that they are no longer 
				 * 			able to receive messages directly from this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				void DisconnectAll(void);

				/**********************************************************************************************//**
				 * @brief	Queries if an object is registered as a listener of
				 * 			this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param [in,out]	listener	The object to query for.
				 *
				 * @return	true if it registered, false otherwise.
				 **************************************************************************************************/
				bool IsConnected(MessageListener& listener) const;

			protected:

				/**********************************************************************************************//**
				 * @brief	Dispatches a message through the connected messaging
				 * 			system.
				 *
				 * @author	Renato
				 * @date	27/05/2012
				 *
				 * @param [in,out]	message	The message.
				 **************************************************************************************************/
				void DispatchMessage( Message& message) const;

				/**********************************************************************************************//**
				 * @brief	Returns an iterator object for the listeners connected
				 * 			to this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @return	The iterator object.
				 **************************************************************************************************/
				Iterator GetIterator(void) const;
			};
		}
	}
}


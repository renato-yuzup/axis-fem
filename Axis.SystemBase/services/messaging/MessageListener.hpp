#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/EventMessage.hpp"
#include "services/messaging/ResultMessage.hpp"
#include "services/messaging/filters/MessageFilter.hpp"

#include "foundation/collections/Collectible.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Represents an object that process messages received
			 * 			through a connected messaging system.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	axis::foundation::collections::Collectible
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API MessageListener : public axis::foundation::collections::Collectible
			{
			private:
				axis::services::messaging::filters::MessageFilter *_filter;

				/**********************************************************************************************//**
				 * @brief	Asks this object to process the specified event message.
				 * 			@remark Processing requests are made according to the
				 * 					filter definition of this object. Processing
				 * 					a message does not imply that it will not be
				 * 					forwarded to other objects nor that it will not be
				 * 					stamped.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param [in,out]	volatileMessage	A reference to the received message.
				 **************************************************************************************************/
				virtual void DoProcessEventMessage(EventMessage& volatileMessage);

				/**********************************************************************************************//**
				 * @brief	Asks this object to process the specified result message.
				 * 			@remark Processing requests are made according to the
				 * 					filter definition of this object. Processing
				 * 					a message does not imply that it will not be
				 * 					forwarded to other objects nor that it will not be
				 * 					stamped.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param [in,out]	volatileMessage	A reference to the received message.
				 **************************************************************************************************/
				virtual void DoProcessResultMessage(ResultMessage& volatileMessage);
			public:

				/**********************************************************************************************//**
				 * @brief	Creates a new message listener and initializes it with
				 * 			the default filter definition (no filtering).
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				MessageListener(void);

				/**********************************************************************************************//**
				 * @brief	Creates a new message listener and assigns a clone
				 * 			of the specified filter definition.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	filter	The filter definition.
				 **************************************************************************************************/
				MessageListener(const axis::services::messaging::filters::MessageFilter& filter);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 **************************************************************************************************/
				virtual ~MessageListener(void);

				/**********************************************************************************************//**
				 * @brief	Returns current message filtering definition.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @return	A reference to the current filtering definition object.
				 **************************************************************************************************/
				axis::services::messaging::filters::MessageFilter& GetFilter(void) const;

				/**********************************************************************************************//**
				 * @brief	Replaces current filtering definition by the clone of the
				 * 			specified filter. Old filter objects have resources 
				 * 			automatically freed.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param	filter	The new filtering definition.
				 **************************************************************************************************/
				void ReplaceFilter(const axis::services::messaging::filters::MessageFilter& filter);

				/**********************************************************************************************//**
				 * @brief	Requests this object to process a message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	25 ago 2012
				 *
				 * @param [in,out]	volatileMessage	The message to be processed.
				 **************************************************************************************************/
				void ProcessMessage(Message& volatileMessage);
			};
			
		}
	}
}


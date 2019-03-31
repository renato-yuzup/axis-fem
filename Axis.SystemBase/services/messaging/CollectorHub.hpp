#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/MessageListener.hpp"
#include "services/messaging/CollectorEndpoint.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			/**********************************************************************************************//**
			 * @brief	Represents an object which is able to forward received 
			 * 			messages to other listeners but it is also capable to
			 * 			process this messages accordingly to its needs.
			 * 			@remark These hub objects are essential to build a
			 * 					messaging network which spans into various
			 * 					branches and replicating messages across them.
			 * 			
			 *
			 * @author	Renato T. Yamassaki
			 * @date	24 ago 2012
			 *
			 * @sa	MessageListener
			 * @sa	CollectorEndpoint
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API CollectorHub : public MessageListener, public CollectorEndpoint
			{
			public:

				/**********************************************************************************************//**
				 * @brief	Default constructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				CollectorHub(void);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				virtual ~CollectorHub(void);
			private:

				/**********************************************************************************************//**
				 * @brief	Stamps a received event message with tracing information and 
				 * 			any additional tags so that the final endpoint object 
				 * 			can verify the original message source and the 
				 * 			forwarding nodes.
				 *
				 * @author	Renato
				 * @date	27/05/2012
				 *
				 * @param [in,out]	message	The message to stamp.
				 **************************************************************************************************/
				virtual void StampEventMessage(EventMessage& message) const;

				/**********************************************************************************************//**
				 * @brief	Stamps a received result message with tracing information and 
				 * 			any additional tags so that the final endpoint object 
				 * 			can verify the original message source and the 
				 * 			forwarding nodes.
				 *
				 * @author	Renato
				 * @date	27/05/2012
				 *
				 * @param [in,out]	message	The message to stamp.
				 **************************************************************************************************/
				virtual void StampResultMessage(ResultMessage& message) const;

				/**********************************************************************************************//**
				 * @brief	Asks this object to process the specified event message.
				 * 			@remark Processing requests are made according to the
				 * 					filter definition of this object. Processing
				 * 					a message does not imply that it will not be
				 * 					forwarded to other objects nor that it will not be
				 * 					stamped.
				 * 			@remark It is highly recommended that overrides should
				 * 					not be made for this member. For
				 * 					message processing, the 
				 * 					ProcessEventMessageLocally member should be
				 * 					overridden instead.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param [in,out]	volatileMessage	A reference to the received message.
				 **************************************************************************************************/
				virtual void DoProcessEventMessage( EventMessage& volatileMessage );

				/**********************************************************************************************//**
				 * @brief	Asks this object to process the specified result message.
				 * 			@remark Processing requests are made according to the
				 * 					filter definition of this object. Processing
				 * 					a message does not imply that it will not be
				 * 					forwarded to other objects nor that it will not be
				 * 					stamped.
				 * 			@remark It is highly recommended that overrides should
				 * 					not be made for this member. For
				 * 					message processing, the 
				 * 					ProcessResultMessageLocally member should be
				 * 					overridden instead.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param [in,out]	volatileMessage	A reference to the received message.
				 **************************************************************************************************/
				virtual void DoProcessResultMessage( ResultMessage& volatileMessage );

				/**********************************************************************************************//**
				 * @brief	Process the received event message after it has
				 * 			been stamped by this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	volatileMessage	Message to be processed.
				 **************************************************************************************************/
				virtual void ProcessEventMessageLocally( const EventMessage& volatileMessage );

				/**********************************************************************************************//**
				 * @brief	Process the received result message after it has
				 * 			been stamped by this object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @param	volatileMessage	Message describing the volatile.
				 **************************************************************************************************/
				virtual void ProcessResultMessageLocally( const ResultMessage& volatileMessage );
			};
		}
	}
}

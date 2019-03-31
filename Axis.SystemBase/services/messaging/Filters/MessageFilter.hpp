#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "services/messaging/EventMessage.hpp"
#include "services/messaging/ResultMessage.hpp"

namespace axis
{
	namespace services
	{
		namespace messaging
		{
			namespace filters
			{
				/**********************************************************************************************//**
				 * @brief	Defines a message filter which selectively forwards 
				 * 			for processing messages that comply to a set of
				 * 			requirements.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 **************************************************************************************************/
				class AXISSYSTEMBASE_API MessageFilter
				{
				public:

					/**********************************************************************************************//**
					 * @brief	Destructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					virtual ~MessageFilter(void);

					/**********************************************************************************************//**
					 * @summary	Returns an instance of a default message filter,
					 * 			which forward for processing any message. 
					 * 			@remark  A copy of this object should be used
					 * 					 instead of using it directly, or else
					 * 					 some unexpected behavior might happen
					 * 					 if used simultaneously by message 
					 * 					 listener classes.
					 **************************************************************************************************/
					static const MessageFilter& Default;

					/**********************************************************************************************//**
					 * @brief	Queries if an event message should be filtered.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	message	The message.
					 *
					 * @return	true if the message should be consumed (that is,
					 * 			accepted, but not forwarded for processing), 
					 * 			false otherwise.
					 **************************************************************************************************/
					virtual bool IsEventMessageFiltered(const axis::services::messaging::EventMessage& message) = 0;

					/**********************************************************************************************//**
					 * @brief	Queries if a result message should be filtered.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @param	message	The message.
					 *
					 * @return	true if the message should be consumed (that is,
					 * 			accepted, but not forwarded for processing), 
					 * 			false otherwise.
					 **************************************************************************************************/
					virtual bool IsResultMessageFiltered(const axis::services::messaging::ResultMessage& message) = 0;

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
					virtual MessageFilter& Clone(void) const = 0;
				};			
			}
		}
	}
}


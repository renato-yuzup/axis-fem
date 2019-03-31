#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "MessageFilter.hpp"

namespace axis
{
	namespace services
	{
		 namespace messaging
		 {
			namespace filters
			{
				/**********************************************************************************************//**
				 * @brief	Implements a message filter which accepts any message.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	24 ago 2012
				 *
				 * @sa	MessageFilter
				 **************************************************************************************************/
				class DefaultMessageFilter : public MessageFilter
				{
				public:

					/**********************************************************************************************//**
					 * @brief	Default constructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					DefaultMessageFilter(void);

					/**********************************************************************************************//**
					 * @brief	Destructor.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					~DefaultMessageFilter(void);

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
					virtual bool IsEventMessageFiltered( const axis::services::messaging::EventMessage& message );

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
					virtual bool IsResultMessageFiltered( const axis::services::messaging::ResultMessage& message );

					/**********************************************************************************************//**
					 * @brief	Destroys this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 **************************************************************************************************/
					virtual void Destroy( void ) const;

					/**********************************************************************************************//**
					 * @brief	Makes a deep copy of this object.
					 *
					 * @author	Renato T. Yamassaki
					 * @date	24 ago 2012
					 *
					 * @return	A copy of this object.
					 **************************************************************************************************/
					virtual MessageFilter& Clone( void ) const;
				};
			}
		 }
	}
}

